import numpy as np
import scipy.integrate as integrate
import scipy
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.optimize import root
from scipy import stats
from scipy.optimize import curve_fit, minimize, least_squares
from scipy.special import erf
from scipy.optimize import fsolve

from scipy.integrate import quad, dblquad,tplquad
from scipy.stats import multivariate_normal
from scipy.integrate import solve_ivp

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor,MLPClassifier
import sklearn
from sklearn.model_selection import GridSearchCV
import itertools
import warnings

import pickle

import functions as fun
import sims_utils as su
import network as network
import ricciardi_class as ricciardi_class

warnings.filterwarnings('ignore')

np.set_printoptions(precision=2)

res_param_idxs_fixed,res_param_idxs,res_moment_idxs,res_param_conv_idxs,sim_param_idxs,sim_param_conv_idxs = \
    su.res_param_idxs_fixed,su.res_param_idxs,su.res_moment_idxs,su.res_param_conv_idxs,su.sim_param_idxs,su.sim_param_conv_idxs

moment_idxs = {
    'mean_base_all' : 0,
    'mean_delta_all' : 1,
    'std_base_all' : 2,
    'std_delta_all' : 3,
    'cov_all' : 4,
    'mean_base_tune' : 5,
    'mean_delta_tune' : 6,
    'std_base_tune' : 7,
    'std_delta_tune' : 8,
    'cov_tune' : 9
}

prediction_idxs = {
    'mean_base' : 0,
    'mean_opto' : 1,
    'std_base' : 2,
    'std_opto' : 3,
    'std_delta' : 4,
    'norm_cov' : 5,
}




def build_predictor_funs(sim_params,sim_moments,output_dir,name_in,seedmlp=123,moments=True,mplparams=None):


    # I follow this tutorial https://coderzcolumn.com/tutorials/machine-learning/scikit-learn-sklearn-neural-network
    X = np.copy(sim_params)
    X[:,sim_param_idxs['beta']] = np.log10(X[:,sim_param_idxs['beta']])
    X[:,sim_param_idxs['CV_K']] = np.log10(X[:,sim_param_idxs['CV_K']])
    X[:,sim_param_idxs['CV_Lam']] = np.log10(X[:,sim_param_idxs['CV_Lam']])
    X[:,sim_param_idxs['J']] = np.log10(X[:,sim_param_idxs['J']])
    X[:,sim_param_idxs['rX']] = np.log10(X[:,sim_param_idxs['rX']])
    X[:,sim_param_idxs['L']] = np.log10(X[:,sim_param_idxs['L']])

    Y = np.copy(sim_moments)
    if moments:
        Y[:,moment_idxs['cov_all']] = np.sign(Y[:,moment_idxs['cov_all']])*np.sqrt(np.abs(Y[:,moment_idxs['cov_all']]))
        Y[:,moment_idxs['cov_tune']] = np.sign(Y[:,moment_idxs['cov_tune']])*np.sqrt(np.abs(Y[:,moment_idxs['cov_tune']]))
    else:
        Y[:,prediction_idxs['norm_cov']] = Y[:,prediction_idxs['norm_cov']]*Y[:,prediction_idxs['std_delta']]

    scaler = StandardScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.80, test_size=0.20, random_state=seedmlp)

    # center training dataset
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # apply same transformation to test data
    X_test = scaler.transform(X_test)

    Ch_train = np.any(Y_train > 1e4,1).astype(int)
    Ch_test = np.any(Y_test > 1e4,1).astype(int)

    # Use optimization in the plot reults file to find best network structure
    
    # mlp_classifier  = MLPClassifier(random_state=seedmlp,
    #                            activation='relu',
    #                            hidden_layer_sizes=(100, 150, 50),)
    # mlp_classifier.fit(X_train, Ch_train)
    
    # Use optimization in the plot reults file to find best network structure

    nEns = 5
    mlp_regressors = [None]*nEns
    for i in range(nEns):
        # Use optimization in the plot reults file to find best network structure
        if mplparams is None:
            mlp_regressor  = MLPRegressor(random_state=1000*i+seedmlp,
                                      activation='relu',
                                      hidden_layer_sizes=(100, 150, 50),)
        else:
            mlp_regressor  = MLPRegressor(random_state=1000*i+seedmlp,**mplparams)
        mlp_regressor.fit(X_train[Ch_train == 0], Y_train[Ch_train == 0])
        mlp_regressors[i] = mlp_regressor

#     if mplparams is None:
#         mlp_regressor  = MLPRegressor(random_state=seedmlp,
#                           activation='relu',
#                           hidden_layer_sizes=(100, 150, 50),)
# 
#     else:
# 
#         mlp_regressor  = MLPRegressor(random_state=seedmlp,**mplparams)
#     print(mlp_regressor)
                              
#     mlp_regressor.fit(X_train[Ch_train == 0], Y_train[Ch_train == 0])
    # Y_preds = mlp_regressor.predict(X_test)

    #print(Y_preds[:5])
    #print(Y_test[:5])

    def predict(X):
        predictions = np.zeros((nEns,len(X),6))
        for i in range(nEns):
            predictions[i] = mlp_regressors[i].predict(X)
        return np.mean(predictions,axis=0)

    def score(X,Y):
        u = np.sum((Y - predict(X))**2,axis=0)
        v = np.sum((Y - np.mean(Y,axis=0))**2,axis=0)
        return np.mean(1 - u/v)

#     with open(output_dir+"/mlp_regressor-"+name_in+'.pkl', 'wb') as handle:
#         pickle.dump(mlp_regressor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(output_dir+"/mlp_classifier-"+name_in+'.pkl', 'wb') as handle:
    #     pickle.dump(mlp_classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # with open(output_dir+"/mlp_regressor-"+name_in+'.pkl', 'rb') as handle:
    #     mlp_regressor = pickle.load(handle)
    # with open(output_dir+"/mlp_classifier-"+name_in+'.pkl', 'rb') as handle:
    #     mlp_classifier = pickle.load(handle)

    print('Test R^2 Score : %.3f'%score(X_test[Ch_test == 0], Y_test[Ch_test == 0])) ## Score method also evaluates accuracy for classification models.
    print('Training R^2 Score : %.3f'%score(X_train[Ch_train == 0], Y_train[Ch_train == 0]))
    
    def predictor_sim(X_test):
        X_test_scaled = scaler.transform(X_test)
        # Ch_test = mlp_classifier.predict_proba(X_test_scaled)[:,1]
        Y_test=predict(X_test_scaled)
        # test the point in the classifier, is it a good param and punish it
        return Y_test# + Ch_test*1e3
    
    def predictor_data(inputs,param,nc,animal):
        # Param should be
        # param=[gE,gI,log10(beta),log10(CV_K),SlE,SlI,SoriE,SoriI,Stun,log10(CV_Lam),log10(J)]
        if moments:
            prediction=np.zeros((len(moment_idxs)*nc))
        else:
            prediction=np.zeros((len(prediction_idxs)*nc))
        log10_rX=inputs[0:nc]
        log10_L=inputs[nc]
        X_test=np.zeros((nc,len(sim_param_idxs)))
        for idx in range(len(sim_param_idxs)-2):
            X_test[:,idx]=param[idx]
        X_test[:,sim_param_idxs['rX']]=log10_rX
        X_test[:,sim_param_idxs['L']]=log10_L

        Y_test=predictor_sim(X_test)
        prediction=np.zeros((len(prediction_idxs),nc))

        if moments:
            mean_base_all=Y_test[:,moment_idxs['mean_base_all']]
            mean_delta_all=Y_test[:,moment_idxs['mean_delta_all']]
            std_base_all=Y_test[:,moment_idxs['std_base_all']]
            std_delta_all=Y_test[:,moment_idxs['std_delta_all']]
            cov_base_delta_all=np.sign(Y_test[:,moment_idxs['cov_all']])*Y_test[:,moment_idxs['cov_all']]**2
            mean_base_tune=Y_test[:,moment_idxs['mean_base_tune']]
            mean_delta_tune=Y_test[:,moment_idxs['mean_delta_tune']]
            std_base_tune=Y_test[:,moment_idxs['std_base_tune']]
            std_delta_tune=Y_test[:,moment_idxs['std_delta_tune']]
            cov_base_delta_tune=np.sign(Y_test[:,moment_idxs['cov_tune']])*Y_test[:,moment_idxs['cov_tune']]**2
        
            if animal=='mouse' or animal=='Mouse':
                prediction[prediction_idxs['mean_base'],:]=mean_base_all
                prediction[prediction_idxs['mean_opto'],:]=mean_base_all+mean_delta_all
                prediction[prediction_idxs['std_base'],:]=std_base_all
                prediction[prediction_idxs['std_opto'],:]=np.sqrt(std_base_all**2+std_delta_all**2+2*cov_base_delta_all)
                prediction[prediction_idxs['std_delta'],:]=std_delta_all
                prediction[prediction_idxs['norm_cov'],:]=cov_base_delta_all/std_delta_all**2
            elif animal=='monkey' or animal=='Monkey':
                prediction[prediction_idxs['mean_base'],:]=mean_base_tune
                prediction[prediction_idxs['mean_opto'],:]=mean_base_tune+mean_delta_tune
                prediction[prediction_idxs['std_base'],:]=std_base_tune
                prediction[prediction_idxs['std_opto'],:]=np.sqrt(std_base_tune**2+std_delta_tune**2+2*cov_base_delta_tune)
                prediction[prediction_idxs['std_delta'],:]=std_delta_tune
                prediction[prediction_idxs['norm_cov'],:]=cov_base_delta_tune/std_delta_tune**2
        else:
            mean_base=Y_test[:,prediction_idxs['mean_base']]
            mean_opto=Y_test[:,prediction_idxs['mean_opto']]
            std_base=Y_test[:,prediction_idxs['std_base']]
            std_opto=Y_test[:,prediction_idxs['std_opto']]
            std_delta=Y_test[:,prediction_idxs['std_delta']]
            norm_cov=Y_test[:,prediction_idxs['norm_cov']]/Y_test[:,prediction_idxs['std_delta']]

            prediction[prediction_idxs['mean_base'],:]=mean_base
            prediction[prediction_idxs['mean_opto'],:]=mean_opto
            prediction[prediction_idxs['std_base'],:]=std_base
            prediction[prediction_idxs['std_opto'],:]=std_opto
            prediction[prediction_idxs['std_delta'],:]=std_delta
            prediction[prediction_idxs['norm_cov'],:]=norm_cov

        return prediction

    return predictor_sim,predictor_data

def build_sim_predictor_funs(params_dict_fixed,Omap,RF='in',tuned='yes'):
    ri=ricciardi_class.Ricciardi()
    ri.set_up_nonlinearity()

    seed_con = params_dict_fixed['seed_con']
    KX = params_dict_fixed['KX']
    pmax = params_dict_fixed['pmax']
    SoriE = params_dict_fixed['SoriE']
    Lam = params_dict_fixed['Lam']
    Tmax_over_tau_E = params_dict_fixed['Tmax_over_tau_E']

    T = np.arange(0,1.0*Tmax_over_tau_E*ri.tau_E,ri.tau_I/3);
    mask_time=T>(0.5*Tmax_over_tau_E*ri.tau_E)

    # seeds = [1,3,5,7]
    seeds = [1,5]
    max_min = 10

    sim_cons = {'mouse':  [False,False,False,False,False,False,True],
                'Mouse':  [False,False,False,False,False,False,True],
                'monkey': [False,False,False,False,True,False],
                'Monkey': [False,False,False,False,True,False]}

    def sim_predictor_data(inputs,param,nc,animal):
        GI = param[sim_param_idxs['GI']]
        gE = param[sim_param_idxs['gE']]
        gI = param[sim_param_idxs['gI']]
        beta = 10**param[sim_param_idxs['beta']]
        CV_K = 10**param[sim_param_idxs['CV_K']]
        SlE = param[sim_param_idxs['SlE']]
        SlI = param[sim_param_idxs['SlI']]
        SoriI = param[sim_param_idxs['SoriI']] 
        Stun = param[sim_param_idxs['Stun']]
        CV_Lam = 10**param[sim_param_idxs['CV_Lam']]
        J = 10**param[sim_param_idxs['J']]

        rXs=10**inputs[:-1]
        L=10**inputs[-1]

        if Omap=='map':
            ori_type = 'columnar'
        elif Omap=='sp':
            ori_type = 'saltandpepper'

        prediction=np.zeros((len(prediction_idxs),nc))
        for rX_idx,rX in enumerate(rXs):
            if not sim_cons[animal][rX_idx]:
                continue
            preds = np.zeros((len(seeds),6))
            for seed_idx,seed in enumerate(seeds):
                net = network.network(seed_con=int(seed), n=2, Nl=25, NE=8, gamma=0.25, dl=1,
                                      Sl=np.array([[SlE,SlI],[SlE,SlI]]), Sori=np.array([[SoriE,SoriI],[SoriE,SoriI]]),
                                      Stun=Stun, ori_type=ori_type)
                net.GI = GI
                net.generate_disorder(J,gE,gI,beta,pmax,CV_K,rX,KX,Lam,CV_Lam,0.5,vanilla_or_not=False)
                moments,_,_ = fun.get_moments_of_r_sim(net,ri,T,mask_time,L,RF,tuned,False,max_min)

                mean_base_all=moments[moment_idxs['mean_base_all']]
                mean_delta_all=moments[moment_idxs['mean_delta_all']]
                std_base_all=moments[moment_idxs['std_base_all']]
                std_delta_all=moments[moment_idxs['std_delta_all']]
                cov_base_delta_all=moments[moment_idxs['cov_all']]
                mean_base_tune=moments[moment_idxs['mean_base_tune']]
                mean_delta_tune=moments[moment_idxs['mean_delta_tune']]
                std_base_tune=moments[moment_idxs['std_base_tune']]
                std_delta_tune=moments[moment_idxs['std_delta_tune']]
                cov_base_delta_tune=moments[moment_idxs['cov_tune']]
                
                if animal=='mouse' or animal=='Mouse':
                    preds[seed_idx,prediction_idxs['mean_base']]=mean_base_all
                    preds[seed_idx,prediction_idxs['mean_opto']]=mean_base_all+mean_delta_all
                    preds[seed_idx,prediction_idxs['std_base']]=std_base_all
                    preds[seed_idx,prediction_idxs['std_opto']]=np.sqrt(std_base_all**2+std_delta_all**2+2*cov_base_delta_all)
                    preds[seed_idx,prediction_idxs['std_delta']]=std_delta_all
                    preds[seed_idx,prediction_idxs['norm_cov']]=cov_base_delta_all/std_delta_all**2
                elif animal=='monkey' or animal=='Monkey':
                    preds[seed_idx,prediction_idxs['mean_base']]=mean_base_tune
                    preds[seed_idx,prediction_idxs['mean_opto']]=mean_base_tune+mean_delta_tune
                    preds[seed_idx,prediction_idxs['std_base']]=std_base_tune
                    preds[seed_idx,prediction_idxs['std_opto']]=np.sqrt(std_base_tune**2+std_delta_tune**2+2*cov_base_delta_tune)
                    preds[seed_idx,prediction_idxs['std_delta']]=std_delta_tune
                    preds[seed_idx,prediction_idxs['norm_cov']]=cov_base_delta_tune/std_delta_tune**2
            mask = np.invert(np.any(preds > 1e4,1))
            try:
                mean_preds = np.mean(preds[mask,:],0)
            except:
                mean_preds = np.zero(6)
            mean_preds[np.isnan(mean_preds)==True]=1e4
            prediction[:,rX_idx] = mean_preds

        return prediction

    def sim_predictor_cost(fit_inputs,fit_param,nc,animal,dataset):
        this_prediction=sim_predictor_data(fit_inputs,fit_param,nc,animal)
        this_prediction[np.isnan(this_prediction)==True]=10**10
        this_res=(this_prediction[:,sim_cons[animal]]-dataset[:,sim_cons[animal],0])/dataset[:,sim_cons[animal],1]
        return this_res

    return sim_predictor_data,sim_predictor_cost
    

def fit_inputs_to_data_given_param(dataset,model,param,nc,animal):
    def Residuals_FixedNet(inputs,dataset,model,param,nc,animal):
        # residual for fixed parameters
        prediction=model(inputs,param,nc,animal)
        prediction[np.isnan(prediction)==True]=10**10
        Residuals=(prediction-dataset[:,:,0])/dataset[:,:,1]
        return Residuals.ravel()
    input_min,input_max=-1,2;
    input_0=np.log10(np.linspace(1,5,nc+1))

    #print(input_0)
    res_2 = least_squares(Residuals_FixedNet, input_0,
                          args=(dataset,model,param,nc,animal),
                      bounds=(input_min, input_max))
    inputs=res_2.x
    return inputs


def fit_model_to_data(dataset,model,nc,nRep,animal,cost_model=None,sim_params=None):
    # Param should be
    # param=[gE,gI,log10(beta),log10(CV_K),SlE,SlI,SoriE,SoriI,Stun,log10(CV_Lam),log10(J)]
    
    def Residuals(param,dataset,model,nc,animal,species_list=['mouse','monkey']):
        if animal=='both':
            Residuals=[]
            for idx_species in range(2):
                dataset_idx=dataset[idx_species]
                nc_idx=nc[idx_species]
                animal_idx=species_list[idx_species]
                inputs=fit_inputs_to_data_given_param(dataset_idx,model,param,nc_idx,animal_idx)
                prediction=model(inputs,param,nc_idx,animal_idx)
                prediction[np.isnan(prediction)==True]=10**10
                Residuals_idx=(prediction-dataset_idx[:,:,0])/dataset_idx[:,:,1]
                Residuals=Residuals+[Residuals_idx.ravel()]
            return np.concatenate((Residuals[0],Residuals[1]))
        else:
            # residual for fixed parameters
            inputs=fit_inputs_to_data_given_param(dataset,model,param,nc,animal)
            prediction=model(inputs,param,nc,animal)
            prediction[np.isnan(prediction)==True]=10**10
            Residuals=(prediction-dataset[:,:,0])/dataset[:,:,1]
            
            return Residuals.ravel()

    param_0=np.zeros((nRep,len(sim_param_idxs)-2))

    if sim_params is None:
        param_min=np.asarray([1,3,2,-1,np.log10(3)-4,0,0,20,10,np.log10(3)-1,-5])
        param_max=np.asarray([2,10,10,1,np.log10(3)-1,1,1,60,40,np.log10(3)+1,-2.4])

        GI=su.get_random_model_variable('GI',nRep)
        gE,gI=su.get_random_model_variable(['gE','gI'],nRep)
        beta=su.get_random_model_variable('beta',nRep)
        CV_K=su.get_random_model_variable('CV_K',nRep)
        SlE,SlI=su.get_random_model_variable(['SlE','SlI'],nRep)
        SlI[SlI > 1] = 1
        SoriI=su.get_random_model_variable('SoriI',nRep)
        Stun=su.get_random_model_variable('Stun',nRep)
        CV_Lam=su.get_random_model_variable('CV_Lam',nRep)
        J=su.get_random_model_variable('J',nRep)
    else:
        param_min=0.95*np.min(sim_params[:,:-2],axis=0)
        param_max=1.05*np.max(sim_params[:,:-2],axis=0)

        for param in ['beta','CV_K','CV_Lam','J']:
            param_min[sim_param_idxs[param]] = np.log10(param_min[sim_param_idxs[param]])
            param_max[sim_param_idxs[param]] = np.log10(param_max[sim_param_idxs[param]])

        try:
            param_0_idxs = np.random.choice(len(sim_params),nRep,replace=False)
        except:
            nRep = len(sim_params)
            param_0=np.zeros((nRep,len(sim_param_idxs)-2))
            param_0_idxs = np.random.choice(len(sim_params),nRep,replace=False)

        GI=sim_params[param_0_idxs,sim_param_idxs['GI']]
        gE=sim_params[param_0_idxs,sim_param_idxs['gE']]
        gI=sim_params[param_0_idxs,sim_param_idxs['gI']]
        beta=sim_params[param_0_idxs,sim_param_idxs['beta']]
        CV_K=sim_params[param_0_idxs,sim_param_idxs['CV_K']]
        SlE=sim_params[param_0_idxs,sim_param_idxs['SlE']]
        SlI=sim_params[param_0_idxs,sim_param_idxs['SlI']]
        SoriI=sim_params[param_0_idxs,sim_param_idxs['SoriI']]
        Stun=sim_params[param_0_idxs,sim_param_idxs['Stun']]
        CV_Lam=sim_params[param_0_idxs,sim_param_idxs['CV_Lam']]
        J=sim_params[param_0_idxs,sim_param_idxs['J']]

    param_0[:,sim_param_idxs['GI']]=GI
    param_0[:,sim_param_idxs['gE']]=gE
    param_0[:,sim_param_idxs['gI']]=gI
    param_0[:,sim_param_idxs['beta']]=np.log10(beta)
    param_0[:,sim_param_idxs['CV_K']]=np.log10(CV_K)
    param_0[:,sim_param_idxs['SlE']]=SlE
    param_0[:,sim_param_idxs['SlI']]=SlI
    param_0[:,sim_param_idxs['SoriI']]=SoriI
    param_0[:,sim_param_idxs['Stun']]=Stun
    param_0[:,sim_param_idxs['CV_Lam']]=np.log10(CV_Lam)
    param_0[:,sim_param_idxs['J']]=np.log10(J)
    
    sol=np.zeros((nRep,len(sim_param_idxs)-2))
    cost=np.zeros(nRep)
    for idx_rep in range(nRep):
        with np.printoptions(precision=3, suppress=True):
            print('rep=',idx_rep,' param init=',param_0[idx_rep,:])
        res_2 = least_squares(Residuals, param_0[idx_rep,:],
                              args=(dataset,model,nc,animal),
                              bounds=(param_min, param_max))
        sol[idx_rep,:],cost[idx_rep]=res_2.x,res_2.cost
        fit_param = res_2.x
        if animal=='both':
            res=[]
            if cost_model is None:
                def cost_model(fit_inputs,fit_param,nc,animal,dataset):
                    this_prediction=model(fit_inputs,fit_param,nc,animal)
                    this_prediction[np.isnan(this_prediction)==True]=10**10
                    this_res_idx=(this_prediction[:,::2]-dataset_idx[:,::2,0])/dataset_idx[:,::2,1]
                    return this_res_idx
            for idx_species in range(2):
                dataset_idx=dataset[idx_species]
                nc_idx=nc[idx_species]
                animal_idx=species_list[idx_species]
                fit_inputs=fit_inputs_to_data_given_param(dataset_idx,model,fit_param,nc_idx,animal_idx)
                res_idx=cost_model(fit_inputs,fit_param,nc_idx,animal_idx,dataset_idx)
                res=res+[res_idx.ravel()]
            res = np.concatenate((res[0],res[1]))
        else:
            if cost_model is None:
                def cost_model(fit_inputs,fit_param,nc,animal,dataset):
                    this_prediction=model(fit_inputs,fit_param,nc,animal)
                    this_prediction[np.isnan(this_prediction)==True]=10**10
                    this_res=(this_prediction[:,::3]-dataset[:,::3,0])/dataset[:,::3,1]
                    return this_res
            # residual for fixed parameters
            fit_inputs=fit_inputs_to_data_given_param(dataset,model,fit_param,nc,animal)
            res=cost_model(fit_inputs,fit_param,nc,animal,dataset)
            res = res.ravel()
        cost[idx_rep] = 0.5*np.sum(res**2)
        with np.printoptions(precision=3, suppress=True):
            print(res_2.x,cost[idx_rep])

    return sol,cost

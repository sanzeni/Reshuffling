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
from sklearn.neural_network import MLPRegressor
import sklearn
from sklearn.model_selection import GridSearchCV
import itertools
import warnings

warnings.filterwarnings('ignore')

np.set_printoptions(precision=2)



# Build dataset given the specie: 
#compute moments of response and put them in a vector of length 5*Con
# made of mean rate, mean delta rate, std rate, std delta rate, \sqrt{cov}
N_stat=10**3;

def build_dataset(data):
    Con=np.unique(data[:,1])
    Las=np.unique(data[:,2])
    cells_id=np.unique(data[:,0]);
    Las=[Las[0],Las[-1]]

    Cell_Resp=np.zeros((len(cells_id),len(Con),len(Las),))
    for idx_cell in range(len(cells_id)):
        for idx_con in range(len(Con)):
            for idx_las in range(len(Las)):
                mask=(data[:,0]==cells_id[idx_cell])&(data[:,2]==Las[idx_las])&(data[:,1]==Con[idx_con])
                Trial_Resp=data[mask,3::]
                Cell_Resp[idx_cell,idx_con,idx_las]=np.mean(Trial_Resp[np.isnan(Trial_Resp)==False])

    Bootstrap_idx=np.random.choice(np.arange(len(cells_id)),size=(len(cells_id),N_stat), replace=True)
    Bootstrap_Resp=np.zeros((N_stat,len(cells_id),len(Con),len(Las)))
    for idx_rep in range(N_stat):
        for idx_con in range(len(Con)):
            for idx_las in range(len(Las)):
                Bootstrap_Resp[idx_rep,:,idx_con,idx_las]= Cell_Resp[[Bootstrap_idx[:,idx_rep]],idx_con,idx_las]

    Moments=np.zeros((6,len(Con),2))
    for idx_cases in range(6):
        if (idx_cases==0)|(idx_cases==1):
            # mean  rates
            idx_las=idx_cases;
            Measurements=np.mean(Bootstrap_Resp[:,:,:,idx_las],axis=1)
        if (idx_cases==2)|(idx_cases==3):
            # std  rates
            idx_las=idx_cases-2;
            Measurements=np.std(Bootstrap_Resp[:,:,:,idx_las],axis=1)
        if (idx_cases==4):
            # mean Delta rates
            Delta=Bootstrap_Resp[:,:,:,-1]-Bootstrap_Resp[:,:,:,0]
            Measurements=np.std(Delta,axis=1)
        if (idx_cases==5):
            # rho rates Delta rates
            Base=Bootstrap_Resp[:,:,:,0]
            Delta=Bootstrap_Resp[:,:,:,-1]-Bootstrap_Resp[:,:,:,0]
            Measurements=np.zeros((N_stat,len(Con)))
            for idx_rep in range(N_stat):
                for idx_con in range(len(Con)): 
                    pippo=np.cov(Base[idx_rep,:,idx_con],Delta[idx_rep,:,idx_con])
                    Measurements[idx_rep,idx_con]=pippo[0,1]/pippo[1,1]
        
        for idx_con in range(len(Con)): 
            Moments[idx_cases,idx_con,:]=np.mean(Measurements[:,idx_con]),np.std(Measurements[:,idx_con])

    return Moments,Con,len(Con)

def build_function(results):
    # I follow this tutorial https://coderzcolumn.com/tutorials/machine-learning/scikit-learn-sklearn-neural-network
    sim_param_all=results[:,0:15]
    moments_of_r_sim_all=results[:,15:20]
    sim_convergence_all=results[:,20:24]
    sim_decay_all=results[:,26]/results[:,27]

    sim_g_E=(sim_param_all[:,2])
    sim_g_I=(sim_param_all[:,3])
    sim_beta=(sim_param_all[:,4])
    sim_CV_K=(sim_param_all[:,7])
    sim_sigma_Lambda_over_Lambda=(sim_param_all[:,10])
    sim_J=(sim_param_all[:,11])
    sim_r_X=(sim_param_all[:,12])
    sim_ell=(sim_param_all[:,13])

    
    scaler = StandardScaler()  

    X=np.vstack((sim_g_E,sim_g_I,sim_beta,sim_CV_K,sim_sigma_Lambda_over_Lambda,
                 sim_J,sim_r_X,sim_ell)).T
    X[:,2::]=np.log10(X[:,2::])

    Y=np.zeros(np.shape(moments_of_r_sim_all))
    Y[:,:]=moments_of_r_sim_all[:,:]
    Y[:,4]=np.sign(Y[:,4])*np.sqrt(np.abs(Y[:,4]))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.80, test_size=0.20, random_state=123)

    # center training dataset
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    # apply same transformation to test data
    X_test = scaler.transform(X_test)
    
    # Use optimization in the plot reults file to find best network structure
    mlp_regressor  = MLPRegressor(random_state=123,
                              activation='relu',
                              hidden_layer_sizes=(100, 150, 50),)
    mlp_regressor.fit(X_train, Y_train)

    Y_preds = mlp_regressor.predict(X_test)

    #print(Y_preds[:5])
    #print(Y_test[:5])

    print('Test R^2 Score : %.3f'%mlp_regressor.score(X_test, Y_test)) ## Score method also evaluates accuracy for classification models.
    print('Training R^2 Score : %.3f'%mlp_regressor.score(X_train, Y_train))
    
    def Predictor_sim(X_test):
        X_test_scaled = scaler.transform(X_test)
        Y_test=mlp_regressor.predict(X_test_scaled)
        return Y_test
    
    def Predictor_data(inputs,param,nCon,):
        prediction=np.zeros((5*nCon))
        log10_r_X=inputs[0:nCon]
        log10_ell=inputs[nCon]
        X_test=np.zeros((nCon,8))
        for idx in range(6):
            X_test[:,idx]=param[idx]

        X_test[:,6]=log10_r_X
        X_test[:,7]=log10_ell

        Y_test=Predictor_sim(X_test)
        prediction=np.zeros((6,nCon))

        mean_Base=Y_test[:,0]
        mean_Delta=Y_test[:,1]
        std_Base=Y_test[:,2]
        std_Delta=Y_test[:,3]
        cov_Base_Delta=np.sign(Y_test[:,4])*Y_test[:,4]**2
        
        prediction[0,:]=mean_Base
        prediction[1,:]=mean_Base+mean_Delta
        prediction[2,:]=std_Base
        prediction[3,:]=np.sqrt(std_Base**2+std_Delta**2+2*cov_Base_Delta)
        prediction[4,:]=std_Delta
        prediction[5,:]=cov_Base_Delta/std_Delta**2
        return prediction

    return Predictor_sim,Predictor_data
    

def fit_inputs_to_data_given_param(dataset,model,param,nCon):
    def Residuals_FixedNet(inputs,dataset,model,param,nCon,):
        # residual for fixed parameters
        prediction=model(inputs,param,nCon,)
        prediction[np.isnan(prediction)==True]=10**10
        Residuals=(prediction-dataset[:,:,0])/dataset[:,:,1]
        return Residuals.ravel()
    input_min,input_max=-1,2;
    input_0=np.log10(np.linspace(1,5,nCon+1))

    #print(input_0)
    res_2 = least_squares(Residuals_FixedNet, input_0,
                          args=(dataset,model,param,nCon,),
                      bounds=(input_min, input_max))
    inputs=res_2.x
    return inputs


def fit_model_to_data(dataset,model,nCon,nRep):
    # Param should be 
    # param=[g_E,g_I,np.log10(beta),np.log10(sigma_Lambda_over_Lambda),np.log10(J),np.log10(CV_K)]

    
    def Residuals(param,dataset,model,nCon,):
        # residual for fixed parameters
        inputs=fit_inputs_to_data_given_param(dataset,model,param,nCon)    
        prediction=model(inputs,param,nCon,)
        prediction[np.isnan(prediction)==True]=10**10      
        Residuals=(prediction-dataset[:,:,0])/dataset[:,:,1]
        
        return Residuals.ravel()
    
    # sim_g_E,sim_g_I,sim_beta,sim_CV_K,
    
    param_min=np.asarray([3,2,-1,np.log10(3*10**(0*3-4)),-1,-5])
    param_max=np.asarray([10,10,1,np.log10(3*10**(1*3-4)),1,-2.4])
    
    g_E=np.random.rand(nRep)*7+3
    g_I=np.random.rand(nRep)*(g_E-2.5)+2
    beta=10**(np.random.rand(nRep)*1-1)
    CV_K=3*10**(np.random.rand()*3-4)
    sigma_Lambda_over_Lambda=10**(np.random.rand(nRep)*2-1)
    J=10**(np.random.rand(nRep)*2-5)
    
    param_0=np.zeros((nRep,6))
    param_0[:,0]=g_E
    param_0[:,1]=g_I
    param_0[:,2]=np.log10(beta)
    param_0[:,3]=np.log10(CV_K)
    param_0[:,4]=np.log10(sigma_Lambda_over_Lambda)
    param_0[:,5]=np.log10(J)
        
    sol=np.zeros((nRep,6))
    cost=np.zeros(nRep)
    for idx_rep in range(nRep):
        print('rep=',idx_rep,' param init=',param_0[idx_rep,:])
        res_2 = least_squares(Residuals, param_0[idx_rep,:],
                          args=(dataset,model,nCon,),
                          bounds=(param_min, param_max))
        print(res_2.x,res_2.cost)
        sol[idx_rep,:],cost[idx_rep]=res_2.x,res_2.cost

    return sol,cost

def fit_model_to_data_both_species(DATA_both_species,model,nCon,nRep):
    # Param should be 
    # param=[g_E,g_I,np.log10(beta),np.log10(sigma_Lambda_over_Lambda_0),np.log10(sigma_Lambda_over_Lambda_1),np.log10(J)]
    
    dataset_both_species=DATA_both_species[0]
    Con_both_species=DATA_both_species[1]
    nCon_both_species=DATA_both_species[2]
    normalization_both_species=DATA_both_species[3]
    
    def Residuals(param_both_species,dataset_both_species,model,nCon_both_species,normalization_both_species):
        Residuals_both_species=[]
        for idx_species in range(2):
            param=param_both_species  
            
            dataset=dataset_both_species[idx_species]
            nCon=nCon_both_species[idx_species]
            normalization=normalization_both_species[idx_species]
            
            inputs=fit_inputs_to_data_given_param(dataset,model,param,nCon)        
            prediction=model(inputs,param,nCon,)
            prediction[np.isnan(prediction)==True]=10**10      
            Residuals=(prediction-dataset[:,:,0])/dataset[:,:,1]/normalization
            Residuals_both_species=Residuals_both_species+[Residuals.ravel()]
        return np.concatenate((Residuals_both_species[0],Residuals_both_species[1]))
    
    param_min=np.asarray([3,2,-1,np.log10(3*10**(0*3-4)),-1,-5])
    param_max=np.asarray([10,10,1,np.log10(3*10**(1*3-4)),1,-2.4])
    
    g_E=np.random.rand(nRep)*7+3
    g_I=np.random.rand(nRep)*(g_E-2.5)+2
    beta=10**(np.random.rand(nRep)*1-1)
    CV_K=3*10**(np.random.rand()*3-4)
    sigma_Lambda_over_Lambda=10**(np.random.rand(nRep)*2-1)
    J=10**(np.random.rand(nRep)*2-5)
      
    param_0=np.zeros((nRep,6))
    param_0[:,0]=g_E
    param_0[:,1]=g_I
    param_0[:,2]=np.log10(beta)
    param_0[:,3]=np.log10(CV_K)
    param_0[:,4]=np.log10(sigma_Lambda_over_Lambda)
    param_0[:,5]=np.log10(J)
            
    sol=np.zeros((nRep,6))
    cost=np.zeros(nRep)
    for idx_rep in range(nRep):
        print('rep=',idx_rep,' param init=',param_0[idx_rep,:])
        res_2 = least_squares(Residuals, param_0[idx_rep,:],
                          args=(dataset_both_species,model,nCon_both_species,normalization_both_species),
                          bounds=(param_min, param_max))
        print(res_2.x,res_2.cost)
        sol[idx_rep,:],cost[idx_rep]=res_2.x,res_2.cost

    return sol,cost


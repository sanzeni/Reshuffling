#!/usr/bin/python
import sys
import os
import argparse
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import gc

from scipy.interpolate import interp1d

import network as network
import functions as fun
import time


import ricciardi_class as ricciardi_class
ri=ricciardi_class.Ricciardi()

def get_random_model_variable(this_variable,nrep=1):

    if not isinstance(this_variable, list):
        this_variable_iter=[this_variable]
    else:
        this_variable_iter=this_variable
    output=[]
    for k in this_variable_iter:
        if k=='J':
            J=10**(np.random.rand(nrep)*2-5)
            output.append(J)
        elif k=='gE':
            gE=np.random.rand(nrep)*7+3
            output.append(gE)
        elif k=='gI':
            gI=np.random.rand(nrep)*(gE-2.5)+2
            output.append(gI)
        elif k=='beta':
            beta=10**(np.random.rand(nrep)*2-1)
            output.append(beta)
        elif k=='CV_K':
            CV_K=3*10**(np.random.rand(nrep)*3-4)
            output.append(CV_K)
        elif k=='rX':
            rX=5*10**(np.random.rand(nrep)*2-1)
            output.append(rX)
        elif k=='L':
            L=5*10**(np.random.rand(nrep)*2-1)
            output.append(L)
        elif k=='CV_Lam':
            CV_Lam=3*10**(np.random.rand(nrep)*2-1)
            output.append(CV_Lam)
        else:
            print('Your input variable ' + k +'is not in this list')
            return None
    if not isinstance(this_variable, list):
        if nrep==1:
            return output[0][0]
        else:
            return output[0]
    else:
        if nrep==1:
            flat_out = []
            for item in output:
                if np.isscalar(item):
                    flat_out.append(item)
                else:
                    flat_out.append(item[0])
            return flat_out
        else:
            return output






res_param_idxs_fixed = {
    'K'                 : 5,
    'p'                 : 6,
    'Lam'               : 9,
    'Tmax_over_tau_E'   : 14
}

res_param_idxs = {
    'gE'    : 2,
    'gI'    : 3,
    'beta'  : 4,
    'CV_K'  : 7,
    'CV_Lam': 10,
    'J'     : 11,
    'rX'    : 12,
    'L'     : 13
}


res_moment_idxs = {
    'mean_base_all' : 15,
    'mean_delta_all' : 16,
    'std_base_all' : 17,
    'std_delta_all' : 18,
    'cov_all' : 19
}

res_param_conv_idxs = {
    'std_time_diff' : 20,
    'max_time_diff' : 21,
    'std_time_diff_over_sum' : 22,
    'max_time_diff_over_sum' : 23,
    'pippo_E_01over11' : 24,
    'pippo_I_01over11' : 25,
    'tau_decay' : 26,
    'max_decay' : 27
}

sim_param_idxs = {
    'gE' : 0,
    'gI' : 1,
    'beta' : 2,
    'CV_K' : 3,
    'CV_Lam' : 4,
    'J' : 5,
    'rX' : 6,
    'L' : 7
}

sim_param_conv_idxs = {
    'std_time_diff' : 0,
    'max_time_diff' : 1,
    'std_time_diff_over_sum' : 2,
    'max_time_diff_over_sum' : 3,
    'pippo_E_01over11' : 4,
    'pippo_I_01over11' : 5,
    'tau_decay' : 6,
    'max_decay' : 7
}
moment_idxs = {
    'mean_base_all' : 0,
    'mean_delta_all' : 1,
    'std_base_all' : 2,
    'std_delta_all' : 3,
    'cov_all' : 4
}

prediction_idxs = {
    'mean_base' : 0,
    'mean_opto' : 1,
    'std_base' : 2,
    'std_opto' : 3,
    'std_delta' : 4,
    'norm_cov' : 5,
}

#
#    r_convergence = np.asarray([np.std(pippo_m),np.max(np.abs(pippo_m)),
#                               np.std(pippo_n),np.max(np.abs(pippo_n))])
#    r_pippo = np.asarray([pippo_E[0,1]/pippo_E[1,1],pippo_I[0,1]/pippo_I[1,1],tau_decay,max_decay])
#
#

    
def return_filtered_results(results,filter):

    if filter=='filter_td':
        ss_td=results[:,sim_param_conv_idxs['std_time_diff_over_sum']]<0.25
        print(' Time diff filter restricted to '+ str(np.around(sum(ss_td)/len(ss_td)*100))+ ' % of data')

        return results[ss_td,:]

    elif filter=='filter_tau':
        ss_tau=results[:,sim_param_conv_idxs['tau_decay']]>2.99
        print(' Tau filter restricted to '+ str(np.around(sum(ss_tau)/len(ss_tau)*100))+ ' % of data')
        return results[ss_tau,:]
    else:
        return results



def get_concatenated_results(path2results):
    results_file='results_structless'
    print('Your results files are of the form: ' +results_file)
    file_idxs = []
    for file in os.listdir(path2results):
        if file.startswith(results_file):
            file_idxs.append(int(file.replace(results_file+'_','').replace('.txt','')))
    file_idxs.sort()
    print('Found '+str(len(file_idxs))+' results files')
    resultspre=path2results+results_file
    init=True
    results = None
    for i in file_idxs:
        this_results_file=resultspre+'_'+str(i)+'.txt'
#        print('loading ' +this_results_file )
        this_loaded_results=np.loadtxt(this_results_file)
        if init:
            results = this_loaded_results
            init = False
        else:
            results = np.concatenate((results,this_loaded_results))
    jn=file_idxs[-1]
    print('Last loaded file number: ' + str(jn))
    return results, jn

def get_params_dict_best_fit(sim_params_fit,params_dict_fixed):
    X = np.copy(sim_params_fit)
    X[sim_param_idxs['beta']]   = 10**(X[sim_param_idxs['beta']])
    X[sim_param_idxs['CV_K']]   = 10**(X[sim_param_idxs['CV_K']])
    X[sim_param_idxs['CV_Lam']] = 10**(X[sim_param_idxs['CV_Lam']])
    X[sim_param_idxs['J']]      = 10**(X[sim_param_idxs['J']])

    params_dict_fit={k:X[sim_param_idxs[k]] for k in list(sim_param_idxs.keys())[:-2]}
    out_dict={**params_dict_fixed, **params_dict_fit}
    return out_dict




def get_sims_best_fit(stim_size,best_params_dict,best_inputs,resultsdir,this_jn):

    name_end='structless'
    name_results='results_'+name_end+'_'+str(this_jn)+'.txt'
    this_results=resultsdir+name_results
    ##################################################################
    # This command makes all the variables in a dict local variables only on the notebook
    # locals().update(best_params_dict)
    
    # This does the same inside a funciton. Super hacky. Work only if the funciton is in the notebook =(
    # sys._getframe(1).f_locals.update(best_params_dict)
    
    # By hand =(
    seed_con =       best_params_dict['seed_con']
    K =              best_params_dict['K']
    p =              best_params_dict['p']
    Lam =            best_params_dict['Lam']
    Tmax_over_tau_E =best_params_dict['Tmax_over_tau_E']
    gE =             best_params_dict['gE']
    gI =             best_params_dict['gI']
    beta =           best_params_dict['beta']
    CV_K =           best_params_dict['CV_K']
    CV_Lam =         best_params_dict['CV_Lam']
    J =              best_params_dict['J']
    
    rX=10**best_inputs[:-1]
    L=10**best_inputs[-1]
    print("Parameters used seed= {:d} // gE= {:.2f} // gI= {:.2f} // beta= {:.2f} // K= {:d} // p= {:.2f} // CV_K= {:.4f}" \
    .format(seed_con,gE,gI,beta,K,p,CV_K))
    print("Lam= {:.3f} // CV_Lam= {:.2f} // J= {:.6f} // rX= {:.2f} // L= {:.2f} // Tmax_over_tau_E= {:d}"\
    .format(Lam,CV_Lam,J,rX,L,Tmax_over_tau_E))
    print('')
    ##################################################################

    ######### Set up Ricciardi
    ri.set_up_nonlinearity()

    ######### Hardoced stuff that we should fix
#    T=np.arange(0,Tmax_over_tau_E*ri.tau_E,ri.tau_I/3);
#    mask_time=T>(10*ri.tau_E)
    T=np.arange(0,1.5*Tmax_over_tau_E*ri.tau_E,ri.tau_I/3);
    mask_time=T>(0.5*Tmax_over_tau_E*ri.tau_E)

    seeds=[1,3,5,7]#np.round(np.random.rand(4)*2**32)
    max_min=10

    start = time.process_time()

    ori_type = 'saltandpepper'
    net=network.network(seed_con=0, n=2, Nl=25, NE=8, gamma=0.25, dl=1, ori_type=ori_type)


    moments_of_r_sim=np.zeros((len(rX),10))
    predictions=np.zeros((len(rX),12))
    RATES={}
    MUES={}
    MUIS={}
    MULS={}
    ##################################################################

    #rX is a vector with contrast values up to nc and then a single laser value
    for this_c in range(len(rX)):
        print('Contrast ' +str(this_c) +' out of '+str(len(rX)))

        start = time.process_time()

        preds= np.zeros((len(seeds),12))
        moments = np.zeros((len(seeds),10))
        rates = np.zeros((len(seeds),2,net.N))
        muEs = np.zeros((len(seeds),net.N))
        muIs = np.zeros((len(seeds),net.N))
        muLs = np.zeros((len(seeds),net.N))
        convs = np.zeros((len(seeds),len(sim_param_conv_idxs)//2))
        pippos = np.zeros((len(seeds),len(sim_param_conv_idxs)//2))

        for seed_idx,seed in enumerate(seeds):
            net=network.network(seed_con=seed, n=2, Nl=25, NE=8, gamma=0.25, dl=1, ori_type=ori_type)
        # if this_c==0:
            net.generate_disorder(J,gE,gI,beta,p,CV_K,rX[this_c],K,Lam,CV_Lam,stim_size,vanilla_or_not=True)
        # else:
        #     net.get_disorder_input(J,gE,gI,beta,rX[this_c],K,stim_size,vanilla_or_not=True)

            moments[seed_idx,:],convs[seed_idx,:],pippos[seed_idx,:],rates[seed_idx,:,:],muEs[seed_idx,:],\
                muIs[seed_idx,:],muLs[seed_idx,:] =\
                fun.get_moments_of_r_sim(net,ri,T,mask_time,L,True,max_min)
            preds[seed_idx,0] = moments[seed_idx,moment_idxs['mean_base_all']]
            preds[seed_idx,1] = moments[seed_idx,moment_idxs['mean_base_all']] + moments[seed_idx,moment_idxs['mean_delta_all']]
            preds[seed_idx,2] = moments[seed_idx,moment_idxs['std_base_all']]
            preds[seed_idx,3] = np.sqrt(moments[seed_idx,moment_idxs['std_base_all']]**2 +\
                                             moments[seed_idx,moment_idxs['std_delta_all']]**2 +\
                                             2*moments[seed_idx,moment_idxs['cov_all']])
            preds[seed_idx,4] = moments[seed_idx,moment_idxs['std_delta_all']]
            preds[seed_idx,5] = moments[seed_idx,moment_idxs['cov_all']] / moments[seed_idx,moment_idxs['std_delta_all']]**2
            preds[seed_idx,6] = moments[seed_idx,moment_idxs['mean_base_tune']]
            preds[seed_idx,7] = moments[seed_idx,moment_idxs['mean_base_tune']] + moments[seed_idx,moment_idxs['mean_delta_tune']]
            preds[seed_idx,8] = moments[seed_idx,moment_idxs['std_base_tune']]
            preds[seed_idx,9] = np.sqrt(moments[seed_idx,moment_idxs['std_base_tune']]**2 +\
                                             moments[seed_idx,moment_idxs['std_delta_tune']]**2 +\
                                             2*moments[seed_idx,moment_idxs['cov_tune']])
            preds[seed_idx,10] = moments[seed_idx,moment_idxs['std_delta_tune']]
            preds[seed_idx,11] = moments[seed_idx,moment_idxs['cov_tune']] / moments[seed_idx,moment_idxs['std_delta_tune']]**2

        mask = np.invert(np.any(moments > 1e4,1))
        try:
            moments_of_r_sim[this_c,:] = np.mean(moments[mask,:],0)
            predictions[this_c,:] = np.mean(preds[mask,:],0)
        except:
            moments_of_r_sim[this_c,:] = np.ones(len(res_moment_idxs))*1e6
            predictions[this_c,:] = np.ones(len(res_moment_idxs))*1e6
        moments_of_r_sim[this_c,:][np.isnan(moments_of_r_sim[this_c,:])==True]=1e6
        predictions[this_c,:][np.isnan(predictions[this_c,:])==True]=1e6
        r_convergence = np.mean(convs,0)
        r_pippo = np.mean(pippos,0)
        RATES[this_c]=np.hstack([rates[i,:,:] for i in np.arange(len(seeds))[mask]])
        MUES[this_c]=np.hstack([muEs[i,:] for i in np.arange(len(seeds))[mask]])
        MUIS[this_c]=np.hstack([muIs[i,:] for i in np.arange(len(seeds))[mask]])
        MULS[this_c]=np.hstack([muLs[i,:] for i in np.arange(len(seeds))[mask]])

        print('')
        print("ODE integration for this C took ",time.process_time() - start," s")
        print('')

        #------------------------------------------------------------------------------------------------------
        # simulations param + mean results + meaurements of rate convergence
        #------------------------------------------------------------------------------------------------------
        start = time.process_time()
        sim_param=np.asarray([seeds[0],GI,gE,gI,beta,K,p,CV_K,SlE,SlI,SoriE,SoriI,Stun,Lam,CV_Lam,J,rX[this_c],L,Tmax_over_tau_E])
        sim_results=moments_of_r_sim[this_c,:]
        sim_convergence=r_convergence

        additional_measurements=r_pippo
        if this_c==0:
            results=np.zeros((len(rX),len(sim_param)
                              +len(sim_results)
                             +len(sim_convergence)
                             +len(additional_measurements)))

        results[this_c,0:len(sim_param)]=sim_param[:]
        results[this_c,len(sim_param):(len(sim_param)+len(sim_results))]=sim_results
        results[this_c,(len(sim_param)+len(sim_results)):(len(sim_param)+len(sim_results)+len(sim_convergence))]=\
            sim_convergence
        results[this_c,(len(sim_param)+len(sim_results)+len(sim_convergence))::]=additional_measurements

        mask_rep=results[:,0]>0
        
        #------------------------------------------------------------------------------------------------------
        # save
        #------------------------------------------------------------------------------------------------------
        
        # Clean file to print results
        f_handle = open(this_results,'w')
        np.savetxt(f_handle,results[mask_rep,:],fmt='%.6f', delimiter='\t')
        f_handle.close()
        print("Saving results took ",time.process_time() - start," s")
        print('')
        
    return moments_of_r_sim, RATES, MUES, MUIS, MULS, predictions


def get_simulations_from_best_fit(output_dir, nameout_fits,stim_size=0.5,this_animals=['mouse','monkey'],resultsdir='./../results',this_jn=0):

    with open(output_dir+'/'+nameout_fits+".pkl", 'rb') as handle_loadModel:
        output_fit=pickle.load(handle_loadModel)

    all_moments=[]
    output_sims={}
    for idx in range(2):
        print('Doing sims for ' + this_animals[idx])
        try:
            best_sims_dict_idx=output_fit['best_sims_dict_both']
        except:
            best_sims_dict_idx=output_fit['best_sims_dict_'+this_animals[idx]]

        best_inputs= output_fit['best_inputs_'+this_animals[idx]]
        nameout_sim=nameout_fits

        Omap='sp'
        RF='all'
        tuned='all'
        pred_idxs=np.arange(0,6)
        moments_of_r_sim, RATES, MUES, MUIS, MULS, predictions = get_sims_best_fit(stim_size, best_sims_dict_idx,best_inputs,resultsdir,this_jn)
        nc = len(RATES)
        
        output_sims['best_sims_dict_'+this_animals[idx]]=best_sims_dict_idx
        output_sims['best_input_'+this_animals[idx]]=best_inputs
        output_sims['stim_size_'+this_animals[idx]]=stim_size
        output_sims['moments_of_r_sim_'+this_animals[idx]]=moments_of_r_sim
        output_sims['predictions_of_r_sim_'+this_animals[idx]]=predictions[:,pred_idxs]
        output_sims['RATES_'+this_animals[idx]]=RATES
        output_sims['balance_'+this_animals[idx]]=np.array([np.mean(np.abs(MUES[i]+MUIS[i])/MUES[i]) for i in range(nc)])
        output_sims['opto_ratio_'+this_animals[idx]]=np.array([np.mean(MULS[i]/MUES[i]) for i in range(nc)])

    nameout_sim=output_dir+'/Simulation_'+nameout_fits
    with open(nameout_sim+".pkl", 'wb') as handle_Model:
        pickle.dump(output_sims, handle_Model, protocol=pickle.HIGHEST_PROTOCOL)


#    J=10**(np.random.rand()*2-5)
#    gE=np.random.rand()*7+3
#    gI=np.random.rand()*(gE-2.5)+2
#    beta=10**(np.random.rand()*2-1)
#    CV_K=3*10**(np.random.rand()*3-4)
#    rX=5*10**(np.random.rand()*2-1)
#    L=5*10**(np.random.rand()*2-1)
#    CV_Lam=10**(np.random.rand()*2-1)
#    SlE=(np.random.rand())
#    SlI=(np.random.rand())
#    SoriE=30.0
#    SoriI=(np.random.rand()*40+20)
#    Stun=(np.random.rand()*30+10)

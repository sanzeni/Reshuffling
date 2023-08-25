#!/usr/bin/python
import sys
import os
import argparse
import functions as fun
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import gc

from scipy.interpolate import interp1d

import pandas as pd
import network as network
import functions as fun
import functions as funsim
import load_utils as lu
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
        elif k=='GI':
            GI=(np.random.rand(nrep)+1)
            output.append(GI)
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
        elif k=='SlE':
            SlE=(np.random.rand(nrep))
            output.append(SlE)
        elif k=='SlI':
            SlI=(np.random.rand(nrep)*4+2)/3*SlE
            output.append(SlI)
        elif k=='SoriE':
            SoriE=30.0
            output.append(SoriE)
        elif k=='SoriI':
            SoriI=(np.random.rand(nrep)*40+20)
            output.append(SoriI)
        elif k=='Stun':
            Stun=(np.random.rand(nrep)*30+10)
            output.append(Stun)
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
    'seed_con'          : 0,
    'KX'                : 5,
    'pmax'              : 6,
    'SoriE'             : 10,
    'Lam'               : 13,
    'Tmax_over_tau_E'   : 18
}

res_param_idxs = {
    'GI'    : 1,
    'gE'    : 2,
    'gI'    : 3,
    'beta'  : 4,
    'CV_K'  : 7,
    'SlE'   : 8,
    'SlI'   : 9,
    'SoriI' : 11,
    'Stun'  : 12,
    'CV_Lam': 14,
    'J'     : 15,
    'rX'    : 16,
    'L'     : 17
}


res_moment_idxs = {
    'mean_base_all' : 19,
    'mean_delta_all' : 20,
    'std_base_all' : 21,
    'std_delta_all' : 22,
    'cov_all' : 23,
    'mean_base_tune' : 24,
    'mean_delta_tune' : 25,
    'std_base_tune' : 26,
    'std_delta_tune' : 27,
    'cov_tune' : 28
}

res_param_conv_idxs = {
    'std_time_diff' : 29,
    'max_time_diff' : 30,
    'std_time_diff_over_sum' : 31,
    'max_time_diff_over_sum' : 32,
    'pippo_E_01over11' : 33,
    'pippo_I_01over11' : 34,
    'tau_decay' : 35,
    'max_decay' : 36
}

sim_param_idxs = {
    'GI' : 0,
    'gE' : 1,
    'gI' : 2,
    'beta' : 3,
    'CV_K' : 4,
    'SlE' : 5,
    'SlI' : 6,
    'SoriI' : 7,
    'Stun' : 8,
    'CV_Lam' : 9,
    'J' : 10,
    'rX' : 11,
    'L' : 12
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



def get_concatenated_results(path2results,Omap,tuned,RF):
    if map is not None:
        results_file='results_Omap='+str(Omap)+'_RF='+str(RF)+'_Tuned='+str(tuned)
    else:
        results_file='results'
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

def get_params_dict_best_fit_fixed_J(J,sim_params_fit,params_dict_fixed):
    X = np.zeros(11)
    X[:-1] = np.copy(sim_params_fit)
    X[sim_param_idxs['beta']]   = 10**(X[sim_param_idxs['beta']])
    X[sim_param_idxs['CV_K']]   = 10**(X[sim_param_idxs['CV_K']])
    X[sim_param_idxs['CV_Lam']] = 10**(X[sim_param_idxs['CV_Lam']])
    X[sim_param_idxs['J']]      = J

    params_dict_fit={k:X[sim_param_idxs[k]] for k in list(sim_param_idxs.keys())[:-2]}
    out_dict={**params_dict_fixed, **params_dict_fit}
    return out_dict




def get_sims_best_fit(stim_size,best_params_dict,best_inputs,Omap,RF,tuned,resultsdir,this_jn,seed_sorted=False,return_inputs=False,return_rates=False):

    name_end='Omap='+Omap+'_RF='+RF+'_Tuned='+tuned
    name_results='results_'+name_end+'_'+str(this_jn)+'.txt'
    this_results=resultsdir+name_results
    ##################################################################
    # This command makes all the variables in a dict local variables only on the notebook
    # locals().update(best_params_dict)
    
    # This does the same inside a funciton. Super hacky. Work only if the funciton is in the notebook =(
    # sys._getframe(1).f_locals.update(best_params_dict)
    
    # By hand =(
    seed_con =       best_params_dict['seed_con']
    KX =             best_params_dict['KX']
    pmax =           best_params_dict['pmax']
    SoriE =          best_params_dict['SoriE']
    Lam =            best_params_dict['Lam']
    Tmax_over_tau_E =best_params_dict['Tmax_over_tau_E']
    GI =             best_params_dict['GI']
    gE =             best_params_dict['gE']
    gI =             best_params_dict['gI']
    beta =           best_params_dict['beta']
    CV_K =           best_params_dict['CV_K']
    SlE =            best_params_dict['SlE']
    SlI =            best_params_dict['SlI']
    SoriI =          best_params_dict['SoriI']
    Stun =           best_params_dict['Stun']
    CV_Lam =         best_params_dict['CV_Lam']
    J =              best_params_dict['J']
    
    rX=10**best_inputs[:-1]
    L=10**best_inputs[-1]
    print("Parameters used seed= {:d} // GI= {:.2f} // gE= {:.2f} // gI= {:.2f} // beta= {:.2f} // KX= {:d} // pmax= {:.2f}" \
    .format(int(seed_con),GI,gE,gI,beta,int(KX),pmax))
    print("CV_K= {:.4f} // SlE= {:.3f} // SlI= {:.3f} // SoriE= {:.2f} // SoriI= {:.2f} // Stun= {:.2f}"\
    .format(CV_K,SlE,SlI,SoriE,SoriI,Stun))
    print("Lam= {:.3f} // CV_Lam= {:.2f} // J= {:.6f} // rX= {:.2f} // L= {:.2f} // Tmax_over_tau_E= {:d}"\
    .format(Lam,CV_Lam,J,rX[0],L,int(Tmax_over_tau_E)))
    print('')
    ##################################################################

    ######### Set up Ricciardi
    ri.set_up_nonlinearity()

    ######### Hardoced stuff that we should fix
#    T=np.arange(0,Tmax_over_tau_E*ri.tau_E,ri.tau_I/3);
#    mask_time=T>(10*ri.tau_E)
    T=np.arange(0,1.2*Tmax_over_tau_E*ri.tau_E,ri.tau_I/3);
    mask_time=T>(0.2*Tmax_over_tau_E*ri.tau_E)

    seeds=[1,3,5,7]#np.round(np.random.rand(4)*2**32)
    max_min=20

    if Omap=='map':
        ori_type = 'columnar'
    elif Omap=='sp':
        ori_type = 'saltandpepper'

    start = time.process_time()

    params_dict=best_params_dict.copy()

    params_dict['L'] = 10**best_inputs[-1]

    params_dict['Nl']=25
    params_dict['NE']=8
    params_dict['n']=2
    params_dict['gamma']=0.25
    params_dict['dl']=1

    params_dict['ori_type']=ori_type
    params_dict['vanilla_or_not']=False

    params_dict['Stim_Size']=stim_size


    if return_inputs and not return_rates:
        moms,preds_vsm,preds_all,bals,optrs,muXs,muEs,muIs=\
        lu.sim_avg_map(params_dict,rX,ri,T,mask_time,RF,tuned,seeds,max_min,return_inputs,return_rates)
    elif return_rates and not return_inputs:
        moms,preds_vsm,preds_all,bals,optrs,rates=\
        lu.sim_avg_map(params_dict,rX,ri,T,mask_time,RF,tuned,seeds,max_min,return_inputs,return_rates)
    elif return_rates and return_inputs:
        moms,preds_vsm,preds_all,bals,optrs,muXs,muEs,muIs,rates=\
        lu.sim_avg_map(params_dict,rX,ri,T,mask_time,RF,tuned,seeds,max_min,return_inputs,return_rates)
    else:
        moms,preds_vsm,preds_all,bals,optrs=\
        lu.sim_avg_map(params_dict,rX,ri,T,mask_time,RF,tuned,seeds,max_min,return_inputs,return_rates)

#    moms,preds_vsm,preds_all,bals,optrs =\
#        lu.sim_avg_map(params_dict,rX,ri,T,mask_time,RF,tuned,seeds,max_min,return_inputs,return_rates)

    moments_of_r_sim=np.zeros((len(rX),10))
    predictions=np.zeros((len(rX),12))
    balance_indices=np.zeros((len(rX),2))
    opto_ratios=np.zeros((len(rX),2))
    ##################################################################

    #rX is a vector with contrast values up to nc and then a single laser value
    for this_c in range(len(rX)):
        print('Contrast ' +str(this_c) +' out of '+str(len(rX)))

        moments_of_r_sim[this_c] = moms[this_c]
        predictions[this_c] = np.concatenate((preds_all[this_c],preds_vsm[this_c]))
        balance_indices[this_c] = bals[this_c][0:2]
        opto_ratios[this_c] = optrs[this_c][0:2]

        #------------------------------------------------------------------------------------------------------
        # simulations param + mean results + meaurements of rate convergence
        #------------------------------------------------------------------------------------------------------
        # start = time.process_time()
        # sim_param=np.asarray([seeds[0],GI,gE,gI,beta,KX,pmax,CV_K,SlE,SlI,SoriE,SoriI,Stun,Lam,CV_Lam,J,rX[this_c],L,Tmax_over_tau_E])
        # sim_results=moments_of_r_sim[this_c,:]
        # sim_convergence=r_convergence

        # additional_measurements=r_pippo
        # if this_c==0:
        #     results=np.zeros((len(rX),len(sim_param)
        #                       +len(sim_results)
        #                      +len(sim_convergence)
        #                      +len(additional_measurements)))

        # results[this_c,0:len(sim_param)]=sim_param[:]
        # results[this_c,len(sim_param):(len(sim_param)+len(sim_results))]=sim_results
        # results[this_c,(len(sim_param)+len(sim_results)):(len(sim_param)+len(sim_results)+len(sim_convergence))]=\
        #     sim_convergence
        # results[this_c,(len(sim_param)+len(sim_results)+len(sim_convergence))::]=additional_measurements

        # mask_rep=results[:,0]>0
        
        #------------------------------------------------------------------------------------------------------
        # save
        #------------------------------------------------------------------------------------------------------
        
        # Clean file to print results
        # f_handle = open(this_results,'w')
        # np.savetxt(f_handle,results[mask_rep,:],fmt='%.6f', delimiter='\t')
        # f_handle.close()
        # print("Saving results took ",time.process_time() - start," s")
        # print('')
        
        
    if return_inputs and not return_rates:
        return moments_of_r_sim, predictions, balance_indices, opto_ratios, muXs,muEs,muIs

    elif return_rates and not return_inputs:
        return moments_of_r_sim, predictions, balance_indices, opto_ratios, rates
        
    elif return_rates and return_inputs:
        return moments_of_r_sim, predictions, balance_indices, opto_ratios, muXs,muEs,muIs, rates
        
    else:
        return moments_of_r_sim, predictions, balance_indices, opto_ratios





def get_simulations_from_best_fit(output_dir, nameout_fits,stim_size=0.5,this_animals=['mouse','monkey'],resultsdir='./../results',this_jn=0,seed_sorted=False, return_inputs=False, return_rates=False):

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

        try:
            best_inputs= output_fit['best_inputs_'+this_animals[idx]]
        except:
            best_inputs= output_fit['best_input_'+this_animals[idx]]

        nameout_sim=nameout_fits

        if this_animals[idx]=='mouse':
            Omap='sp'
            RF='all'
            tuned='all'
            pred_idxs=np.arange(0,6)
            bal_idx=0
            optr_idx=0
        elif this_animals[idx]=='monkey':
            Omap='map'
            RF='in'
            tuned='yes'
            pred_idxs=np.arange(6,12)
            bal_idx=1
            optr_idx=1
            
                
            
        if return_inputs and not return_rates:
            moments_of_r_sim, predictions, balance_indices, opto_ratios, muXs,muEs,muIs = get_sims_best_fit(stim_size, best_sims_dict_idx,best_inputs,Omap,RF,tuned,resultsdir,this_jn,seed_sorted,return_inputs, return_rates)
        
        elif return_rates and not return_inputs:
            moments_of_r_sim, predictions, balance_indices, opto_ratios, rates = get_sims_best_fit(stim_size, best_sims_dict_idx,best_inputs,Omap,RF,tuned,resultsdir,this_jn,seed_sorted,return_inputs, return_rates)

        elif return_rates and return_inputs:
            moments_of_r_sim, predictions, balance_indices, opto_ratios, muXs,muEs,muIs, rates = get_sims_best_fit(stim_size, best_sims_dict_idx,best_inputs,Omap,RF,tuned,resultsdir,this_jn,seed_sorted,return_inputs, return_rates)
        
        else:
            moments_of_r_sim, predictions, balance_indices, opto_ratios = get_sims_best_fit(stim_size, best_sims_dict_idx,best_inputs,Omap,RF,tuned,resultsdir,this_jn,seed_sorted, return_inputs, return_rates)
        # nc = len(RATES)
        
        output_sims['best_sims_dict_'+this_animals[idx]]=best_sims_dict_idx
        output_sims['best_input_'+this_animals[idx]]=best_inputs
        output_sims['stim_size_'+this_animals[idx]]=stim_size
        output_sims['moments_of_r_sim_'+this_animals[idx]]=moments_of_r_sim
        output_sims['predictions_of_r_sim_'+this_animals[idx]]=predictions[:,pred_idxs]
        if return_rates:
            output_sims['RATES_'+this_animals[idx]]=rates
        if return_inputs:
            output_sims['muXs'+this_animals[idx]]=muXs
            output_sims['muEs'+this_animals[idx]]=muEs
            output_sims['muIs'+this_animals[idx]]=muIs

        output_sims['full_balance_'+this_animals[idx]]=balance_indices
        output_sims['full_opto_ratio_'+this_animals[idx]]=opto_ratios
        output_sims['balance_'+this_animals[idx]]=balance_indices[:,bal_idx]
        output_sims['opto_ratio_'+this_animals[idx]]=opto_ratios[:,optr_idx]

    if return_rates:
        nameout_sim=output_dir+'/Simulation_w_Rates'+nameout_fits

 
    if return_inputs and not return_rates:
        nameout_sim=output_dir+'/Simulation_w_Inputs_'+nameout_fits
    
    elif return_rates and not return_inputs:
        nameout_sim=output_dir+'/Simulation_w_Rates_'+nameout_fits

    elif return_rates and return_inputs:
        nameout_sim=output_dir+'/Simulation_w_Inputs_and_Rates_'+nameout_fits
    
    else:
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

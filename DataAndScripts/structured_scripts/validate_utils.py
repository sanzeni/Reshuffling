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
import time
import sims_utils as su
import data_analysis as da
#import ricciardi_class as ricciardi_class
#ri=ricciardi_class.Ricciardi()

try:
    data=da.Data_MonkeyMouse('both','./../experimental_data')
except:
    data=da.Data_MonkeyMouse('both','./../../../DataAndScripts/experimental_data')

res_param_idxs_fixed = {
    'seed_con'          : 0,
    'KX'                : 1,
    'pmax'              : 2,
    'SoriE'             : 3,
    'Lam'               : 4,
    'Tmax_over_tau_E'   : 5
}

res_param_idxs = {
    'GI'    : 6,
    'gE'    : 7,
    'gI'    : 8,
    'beta'  : 9,
    'CV_K'  : 10,
    'SlE'   : 11,
    'SlI'   : 12,
    'SoriI' : 13,
    'Stun'  : 14,
    'CV_Lam': 15,
    'J'     : 16,
    'rX'    : 17,
    'L'     : 18
}
res_preds_idxs = {
    'mean_base' : 19,
    'mean_opto' : 20,
    'std_base'  : 21,
    'std_opto'  : 22,
    'std_delta' : 23,
    'norm_cov'  : 24
}
res_res_idxs = {
    'mean_base' : 25,
    'mean_opto' : 26,
    'std_base'  : 27,
    'std_opto'  : 28,
    'std_delta' : 29,
    'norm_cov'  : 30
}
res_loss_idxs = {
    'loss' : 31
}



untuned_idxs = {
    'mean_base' : 0,
    'mean_delta' : 1,
    'std_base' : 2,
    'std_delta' : 3,
    'cov' : 4
}
tuned_idxs = {
    'mean_base' : 5,
    'mean_delta' : 6,
    'std_base' : 7,
    'std_delta' : 8,
    'cov' : 9
}

def get_concatenated_validated_param(path2validate,animal,inputic,w_contrast=False,close=False):
    if close=='both' or close=='Both':
        res_nclose, jn_nclose = get_concatenated_validated_param(path2validate,animal,inputic,w_contrast,close=False)
        res_yclose, jn_yclose = get_concatenated_validated_param(path2validate,animal,inputic,w_contrast,close=True)
        return np.vstack((res_nclose,res_yclose)), jn_nclose+jn_yclose
    elif close:
        if w_contrast:
            animal_in_file='Param_w_contrast_close_animal_'+str(animal)
            # animal_in_file='Param_close_animal_'+str(animal)
            contrast_in_file=''

        else:
            animal_in_file='Param_close_animal_'+str(animal)
            contrast_in_file=''

    else:
        if w_contrast:
            animal_in_file='Param_w_contrast_animal_'+str(animal)
            contrast_in_file=''

        else:
            animal_in_file='Param_animal_'+str(animal)
            contrast_in_file=''

    if w_contrast:
        if animal == 'mouse' or animal == 'Mouse':
            nc = data.nc[0]
        elif animal == 'monkey' or animal == 'Monkey':
            nc = data.nc[1]
        res_len = res_loss_idxs['loss'] + nc*14 + 1
    else:
        res_len = res_loss_idxs['loss'] + 1

    ic_in_file='inputic_'+str(inputic)

    file_idxs = []
    for file in os.listdir(path2validate):
        if animal_in_file in file and ic_in_file in file and contrast_in_file in file:
            file_idxs.append(file)
    file_idxs.sort()
    print('Found '+str(len(file_idxs))+' valid files for animal '+animal + ' and contrast ' + str(inputic))
    init=True
    
    for i in file_idxs:
        this_results_file=path2validate+str(i)
        try:
            this_loaded_results=np.loadtxt(this_results_file)
            if init:
                if len(this_loaded_results.shape)<2:
                    results = np.array([this_loaded_results])
                else:
                    results = np.array(this_loaded_results)

                init = False
            else:
                if len(this_loaded_results.shape)<2:
                    new_results = np.array([this_loaded_results])
                else:
                    new_results = np.array(this_loaded_results)
            
                results = np.concatenate((results,new_results),axis=0)
        except:
            print('Something went wrong loading ' +this_results_file )
#
#    if close:
#        results = results[0]

    if len(file_idxs)==0:
        results=np.zeros((0,res_len))
    jn=len(file_idxs)
    return results, jn


def get_best_preds_from_validate_param(path2validate,with_contrast=False,close=False,cut=3,covcut=None):
    n_mom=6
    n_pfx=len(su.res_param_idxs_fixed.values())
    n_pft=n_pfx+len(su.res_param_idxs.values())
    n_pre=n_pft+n_mom
    n_res=n_pre+n_mom
    n_loss=n_res+1



    best_preds={}
    best_preds_w_c={}
    best_params={}

    residuals={}
    residuals_w_c={}
    
    loss={}
    loss_w_c={}

    r_X={}
    for animal,animal_idx in zip(data.this_animals,range(len(data.this_animals))):
        best_preds[animal_idx]=[]
        best_preds_w_c[animal_idx]=[]
        best_params[animal_idx]=[]

        residuals[animal_idx]=[]
        residuals_w_c[animal_idx]=[]
        
        loss[animal_idx]=[]
        loss_w_c[animal_idx]=[]
        
        r_X[animal_idx]=[]

        if with_contrast:
            n_rX=n_loss+data.nc[animal_idx]
            n_preds_w_c=n_rX+data.nc[animal_idx]*n_mom
            n_res_w_c=n_preds_w_c+data.nc[animal_idx]*n_mom
            n_loss_w_c=n_res_w_c+1


        for inputic in range(data.nc[animal_idx]):
            best_preds[animal_idx].append([])
            best_preds_w_c[animal_idx].append([])
            best_params[animal_idx].append([])
            
            residuals[animal_idx].append([])
            residuals_w_c[animal_idx].append([])
            
            loss[animal_idx].append([])
            loss_w_c[animal_idx].append([])

            r_X[animal_idx].append([])

        
            validated_params, jn= get_concatenated_validated_param(path2validate,animal,inputic,with_contrast,close)

            # remove parameters with negative feedforward input
            gamma = 0.25
            GEs = np.ones(len(validated_params))
            GIs = validated_params[:,res_param_idxs['GI']]
            gEs = validated_params[:,res_param_idxs['gE']]
            gIs = validated_params[:,res_param_idxs['gI']]
            Ein = gamma*GIs*gEs-GEs
            Iin = gamma*GIs*gIs-GEs

            pos_FF_in_mask = np.logical_and(Ein>0,Iin>0)

            validated_params = validated_params[pos_FF_in_mask,:]

            params_fixed  = validated_params[0,:n_pfx]
            
            aux_loss=validated_params[:,n_res:n_loss].flatten()

#               this_best_order=np.argsort(aux_loss.flatten())
            if hasattr(cut, "__len__"):
                this_best_order=(aux_loss<cut[animal_idx])
            else:
                this_best_order=(aux_loss<cut)

            best_params[animal_idx][inputic]=validated_params[this_best_order,:n_pft]
            best_preds[animal_idx][inputic]=validated_params[this_best_order,n_pft:n_pre]
            residuals[animal_idx][inputic]=validated_params[this_best_order,n_pre:n_res]
            loss[animal_idx][inputic]= validated_params[this_best_order,n_res:n_loss].flatten()

            if with_contrast:
            
            
                r_x_aux             = validated_params[:,n_loss:n_rX]
                print('shape of r_x_aux is '+str(r_x_aux.shape))
                best_preds_aux_1_w_c  = validated_params[:,n_rX:n_preds_w_c]
                best_preds_aux_2_w_c  = best_preds_aux_1_w_c.reshape(best_preds_aux_1_w_c.shape[0],n_mom,data.nc[animal_idx])
                
                residuals_aux_1_w_c   = validated_params[:,n_preds_w_c:n_res_w_c]
                residuals_aux_2_w_c   = residuals_aux_1_w_c.reshape(residuals_aux_1_w_c.shape[0],n_mom,data.nc[animal_idx])
                
                loss_aux_w_c          = validated_params[:,n_res_w_c:n_loss_w_c].flatten()
                if hasattr(cut, "__len__"):
                    this_best_order_w_c=(loss_aux_w_c<cut[animal_idx])
                else:
                    this_best_order_w_c=(loss_aux_w_c<cut)

                if covcut is not None:
                    cov_dataset = data.bootstrap_moments[animal_idx][5,:,:]
                    if hasattr(covcut, "__len__"):
                        this_best_cov_order = 0.5*np.sum(((best_preds_aux_2_w_c[:,5,:]-cov_dataset[:,0])/cov_dataset[:,1])**2,1) < covcut[animal_idx]
                    else:
                        this_best_cov_order = 0.5*np.sum(((best_preds_aux_2_w_c[:,5,:]-cov_dataset[:,0])/cov_dataset[:,1])**2,1) < covcut
                    this_best_order_w_c = np.logical_and(this_best_order_w_c,this_best_cov_order)

                best_preds_w_c[animal_idx][inputic]=best_preds_aux_2_w_c[this_best_order_w_c,:,:]
                r_X[animal_idx][inputic]=r_x_aux[this_best_order_w_c,:]
                residuals_w_c[animal_idx][inputic]=residuals_aux_2_w_c[this_best_order_w_c,:,:]
                loss_w_c[animal_idx][inputic]=loss_aux_w_c[this_best_order_w_c]
                best_params[animal_idx][inputic]=validated_params[this_best_order_w_c,:n_pft]

#            except:
#                pass

    if not with_contrast:
        return best_preds, best_params, residuals, loss
    if with_contrast:
        return best_preds_w_c, best_params, r_X, residuals_w_c, loss_w_c
                 

def return_predictions_from_Rates_and_idxs(RATES,these_idxs):
    nc=len(RATES)
    
    mean_0     =np.zeros(nc)
    mean_L     =np.zeros(nc)
    std_0      =np.zeros(nc)
    std_L      =np.zeros(nc)
    mean_delta =np.zeros(nc)
    cov_norm   =np.zeros(nc)
    
    for k in range(nc):
        try:
            if RATES[k].shape[-1] < these_idxs.max():
                these_idxs_aux = these_idxs[these_idxs < RATES.shape[-1]]
            else:
                these_idxs_aux = these_idxs
        except:
            if RATES[k].shape[-1] < these_idxs[-1]:
                these_idxs_aux = these_idxs[these_idxs < RATES.shape[-1]]
            else:
                these_idxs_aux = these_idxs
        Base_Sim=RATES[k][0,these_idxs_aux]
        Delta_Sim=(RATES[k][1,these_idxs_aux]-RATES[k][0,these_idxs_aux])
        Laser_Sim=RATES[k][1,these_idxs_aux]

        mean_0[k]=np.nanmean(Base_Sim)
        mean_L[k]=np.nanmean(Laser_Sim)
        std_0[k]=np.nanstd(Base_Sim)
        std_L[k]=np.nanstd(Laser_Sim)
        mean_delta[k]=np.nanstd(Delta_Sim)
        cov_norm[k]=np.cov(Delta_Sim,Base_Sim)[1,0]/np.nanstd(Delta_Sim)**2


    return mean_0,mean_L,std_0,std_L,mean_delta,cov_norm


def get_predictions_from_moments_and_idxs(this_moments,this_idx):

    preds=np.zeros(6)
    preds[0] = this_moments[this_idx['mean_base']]
    preds[1] = this_moments[this_idx['mean_base']] + this_moments[this_idx['mean_delta']]
    preds[2] = this_moments[this_idx['std_base']]
    preds[3] = np.sqrt(this_moments[this_idx['std_base']]**2 +\
                                     this_moments[this_idx['std_delta']]**2 +\
                                     2*this_moments[this_idx['cov']])
    preds[4] = this_moments[this_idx['std_delta']]
    preds[5] = this_moments[this_idx['cov']] / this_moments[this_idx['std_delta']]**2

    return preds



def find_and_save_best_validated_fit(path2validate,path2savedfits):

    
    
    fit_preds,fit_params,fit_rX,_,_ = get_best_preds_from_validate_param(path2validate,
                                                                            with_contrast=True,close='Both',
                                                                            cut=[1.2,0.44],covcut=[1.2,0.44])

    fit_preds = {0:[item for sublist in fit_preds[0] for item in sublist][0],
                 1:[item for sublist in fit_preds[1] for item in sublist][0]}
    fit_params = {0:[item for sublist in fit_params[0] for item in sublist][0],
                  1:[item for sublist in fit_params[1] for item in sublist][0]}
    fit_rX = {0:[item for sublist in fit_rX[0] for item in sublist][0],
              1:[item for sublist in fit_rX[1] for item in sublist][0]}


    params_dict_fixed={k:fit_params[0][res_param_idxs_fixed[k]] for k in list(res_param_idxs_fixed.keys())}
    output_fit={}
    for idx in range(len(data.this_animals)):
        dataset=data.bootstrap_moments[idx]
        contrast=data.contrast[idx]
        nc=data.nc[idx]

        predictions=fit_preds[idx]
        X=fit_params[idx][list(res_param_idxs.values())][:-2]
        X[su.sim_param_idxs['beta']]   = np.log10(X[su.sim_param_idxs['beta']])
        X[su.sim_param_idxs['CV_K']]   = np.log10(X[su.sim_param_idxs['CV_K']])
        X[su.sim_param_idxs['CV_Lam']] = np.log10(X[su.sim_param_idxs['CV_Lam']])
        X[su.sim_param_idxs['J']]      = np.log10(X[su.sim_param_idxs['J']])
        best_sims_dict=su.get_params_dict_best_fit(X,params_dict_fixed)

        output_fit['best_inputs_'+data.this_animals[idx]]=np.log10(np.hstack([fit_rX[idx],
                                                                    fit_params[idx][res_param_idxs['L']]]))
        output_fit['best_param_raw_'+data.this_animals[idx]]=fit_params[idx][list(res_param_idxs.values())][:-2]
        output_fit['predictions_'+data.this_animals[idx]]=fit_preds[idx]
        output_fit['best_sims_dict_'+data.this_animals[idx]]=best_sims_dict
        output_fit['nc']=data.nc

    with open(path2savedfits+'best_fit.pkl', 'wb') as handle_Model:
        pickle.dump(output_fit, handle_Model, protocol=pickle.HIGHEST_PROTOCOL)
    print('')
    print('##########################################################')
    print('Your best fit is saved in ' + path2savedfits+'best_fit.pkl')

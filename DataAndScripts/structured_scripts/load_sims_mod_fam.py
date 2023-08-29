import pickle
import numpy as np

import os

import functions_optimal as funopt
import functions as fun
import data_analysis as da
import sims_utils as su
import validate_utils as vu
import load_utils as lu

import ricciardi_class as ric
import network as network

import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr,spearmanr,kendalltau

path2validate='./validation_results/'

data=da.Data_MonkeyMouse('both','../data')

try:
    data=da.Data_MonkeyMouse('both','./../../../DataAndScripts/experimental_data/')
except:
    data=da.Data_MonkeyMouse('both','./../../DataAndScripts/experimental_data/')

# best_preds_w_c, best_params_fitted,best_rX,residuals_w_c, loss_w_c =\
#     vu.get_best_preds_from_validate_param(path2validate, with_contrast=True,close='Both',cut=[2.5,5],covcut=[1,2.5])
best_preds_w_c, best_params_fitted,best_rX,residuals_w_c, loss_w_c =\
    vu.get_best_preds_from_validate_param(path2validate, with_contrast=True,close=False,cut=[3.5,5.5],covcut=[2,4])
# print(best_params_fitted)

param_sets = [None,None]
rX_sets = [None,None]

param_sets[0] = np.array([item[list(vu.res_param_idxs.values())] for sublist in best_params_fitted[0]
                          for item in sublist])
_,uniq_idxs = np.unique(param_sets[0],axis=0,return_index=True)
param_sets[0] = param_sets[0][uniq_idxs,:]
rX_sets[0] = np.array([item for sublist in best_rX[0] for item in sublist])
rX_sets[0] = rX_sets[0][uniq_idxs]

param_sets[1] = np.array([item[list(vu.res_param_idxs.values())] for sublist in best_params_fitted[1]
                          for item in sublist])
_,uniq_idxs = np.unique(param_sets[1],axis=0,return_index=True)
param_sets[1] = param_sets[1][uniq_idxs,:]
rX_sets[1] = np.array([item for sublist in best_rX[1] for item in sublist])
rX_sets[1] = rX_sets[1][uniq_idxs]

# remove rX
param_sets[0][:,su.sim_param_idxs['rX']] = rX_sets[0][:,0]
param_sets[1][:,su.sim_param_idxs['rX']] = rX_sets[1][:,0]

print(param_sets[0].shape)
print(param_sets[1].shape)

ri = ric.Ricciardi()
ri.set_up_nonlinearity()

seeds=[1,3,5,7]
max_min=15

T = np.arange(0,1.5*200*ri.tau_E,ri.tau_I/3);
mask_time=T>(0.5*200*ri.tau_E)

RF = 'in'
tuned = 'yes'

pred_sets = [None,None]
bal_sets = [None,None]
optr_sets = [None,None]
muX_sets = [None,None]
muE_sets = [None,None]
muI_sets = [None,None]

pred_sets[0] = np.zeros((len(param_sets[0]),6))
bal_sets[0] = np.zeros((len(param_sets[0]),6))
optr_sets[0] = np.zeros((len(param_sets[0]),6))
muX_sets[0] = np.zeros((len(param_sets[0]),6))
muE_sets[0] = np.zeros((len(param_sets[0]),6))
muI_sets[0] = np.zeros((len(param_sets[0]),6))

for i in range(len(param_sets[0])):
    params_dict={}
    params_dict['Stim_Size']=0.5
    params_dict['seed_con']=int(seeds[0])
    params_dict['KX']=500
    params_dict['pmax']=0.09
    params_dict['SoriE']=30
    params_dict['Lam']=1e-3
    params_dict['J']=param_sets[0][i,su.sim_param_idxs['J']]
    params_dict['GI']=param_sets[0][i,su.sim_param_idxs['GI']]
    params_dict['gE']=param_sets[0][i,su.sim_param_idxs['gE']]
    params_dict['gI']=param_sets[0][i,su.sim_param_idxs['gI']]
    params_dict['beta']=param_sets[0][i,su.sim_param_idxs['beta']]
    params_dict['CV_K']=param_sets[0][i,su.sim_param_idxs['CV_K']]
    params_dict['SlE']=param_sets[0][i,su.sim_param_idxs['SlE']]
    params_dict['SlI']=param_sets[0][i,su.sim_param_idxs['SlI']]
    params_dict['SoriI']=param_sets[0][i,su.sim_param_idxs['SoriI']]
    params_dict['Stun']=param_sets[0][i,su.sim_param_idxs['Stun']]
    params_dict['CV_Lam']=param_sets[0][i,su.sim_param_idxs['CV_Lam']]
    params_dict['L']=param_sets[0][i,su.sim_param_idxs['L']]

    params_dict['Nl']=25
    params_dict['NE']=8
    params_dict['n']=2
    params_dict['gamma']=0.25
    params_dict['dl']=1

    params_dict['ori_type']='saltandpepper'
    params_dict['vanilla_or_not']=False
    
    _,_,preds_aux,bals_aux,optrs_aux,muX_aux,muE_aux,muI_aux =\
        lu.sim_avg_map(params_dict,rX_sets[0][i,-1:],ri,T,mask_time,RF,tuned,seeds,max_min,True)
    
    pred_sets[0][i] = preds_aux[-1]
    bal_sets[0][i] = bals_aux[-1]
    optr_sets[0][i] = optrs_aux[-1]
    muX_sets[0][i] = muX_aux[-1]
    muE_sets[0][i] = muE_aux[-1]
    muI_sets[0][i] = muI_aux[-1]

pred_sets[1] = np.zeros((len(param_sets[1]),6))
bal_sets[1] = np.zeros((len(param_sets[1]),6))
optr_sets[1] = np.zeros((len(param_sets[1]),6))
muX_sets[1] = np.zeros((len(param_sets[1]),6))
muE_sets[1] = np.zeros((len(param_sets[1]),6))
muI_sets[1] = np.zeros((len(param_sets[1]),6))

for i in range(len(param_sets[1])):
    params_dict={}
    params_dict['Stim_Size']=0.5
    params_dict['seed_con']=int(seeds[0])
    params_dict['KX']=500
    params_dict['pmax']=0.09
    params_dict['SoriE']=30
    params_dict['Lam']=1e-3
    params_dict['J']=param_sets[1][i,su.sim_param_idxs['J']]
    params_dict['GI']=param_sets[1][i,su.sim_param_idxs['GI']]
    params_dict['gE']=param_sets[1][i,su.sim_param_idxs['gE']]
    params_dict['gI']=param_sets[1][i,su.sim_param_idxs['gI']]
    params_dict['beta']=param_sets[1][i,su.sim_param_idxs['beta']]
    params_dict['CV_K']=param_sets[1][i,su.sim_param_idxs['CV_K']]
    params_dict['SlE']=param_sets[1][i,su.sim_param_idxs['SlE']]
    params_dict['SlI']=param_sets[1][i,su.sim_param_idxs['SlI']]
    params_dict['SoriI']=param_sets[1][i,su.sim_param_idxs['SoriI']]
    params_dict['Stun']=param_sets[1][i,su.sim_param_idxs['Stun']]
    params_dict['CV_Lam']=param_sets[1][i,su.sim_param_idxs['CV_Lam']]
    params_dict['L']=param_sets[1][i,su.sim_param_idxs['L']]

    params_dict['Nl']=25
    params_dict['NE']=8
    params_dict['n']=2
    params_dict['gamma']=0.25
    params_dict['dl']=1

    params_dict['ori_type']='columnar'
    params_dict['vanilla_or_not']=False
    
    _,preds_aux,_,bals_aux,optrs_aux,muX_aux,muE_aux,muI_aux =\
        lu.sim_avg_map(params_dict,rX_sets[1][i,-1:],ri,T,mask_time,RF,tuned,seeds,max_min,True)
    
    pred_sets[1][i] = preds_aux[-1]
    bal_sets[1][i] = bals_aux[-1]
    optr_sets[1][i] = optrs_aux[-1]
    muX_sets[1][i] = muX_aux[-1]
    muE_sets[1][i] = muE_aux[-1]
    muI_sets[1][i] = muI_aux[-1]

param_data = [None,None]

param_labels = [r'$G_I$',r'$g_E$',r'$g_I$',r'$\beta$',r'$CV_K$',r'$S_{l,E}$',r'$S_{l,I}$',r'$S_{ori,I}$',
                r'$S_{tune}$',r'$CV_\lambda$',r'$J$',r'$r_X$',r'$L$']
param_data[0] = pd.DataFrame(param_sets[0],columns=param_labels)
param_data[1] = pd.DataFrame(param_sets[1],columns=param_labels)

pred_data = [None,None]

pred_labels = [r'$r_{base}$',r'$r_{opto}$',r'$\sigma_{r_{base}}$',
                r'$\sigma_{r_{opto}}$',r'$\sigma_{\Delta r}$',r'$\rho$',
               r'$\beta_{E+I,all}$',r'$\beta_{E+I,VSM}$',r'$\beta_{E,all}$',
                r'$\beta_{E,VSM}$',r'$\beta_{I,all}$',r'$\beta_{I,VSM}$',
               r'$OIR_{E+I,all}$',r'$OIR_{E+I,VSM}$',r'$OIR_{E,all}$',
                r'$OIR_{E,VSM}$',r'$OIR_{I,all}$',r'$OIR_{I,VSM}$',
               r'$\mu_{X\to E+I,all}$',r'$\mu_{X\to E+I,VSM}$',r'$\mu_{X\to I,all}$',
                r'$\mu_{X\to I,VSM}$',r'$\mu_{X\to E,all}$',r'$\mu_{X\to E,VSM}$',
               r'$\mu_{E\to E+I,all}$',r'$\mu_{E\to E+I,VSM}$',r'$\mu_{E\to I,all}$',
                r'$\mu_{E\to I,VSM}$',r'$\mu_{E\to E,all}$',r'$\mu_{E\to E,VSM}$',
               r'$\mu_{I\to E+I,all}$',r'$\mu_{I\to E+I,VSM}$',r'$\mu_{I\to I,all}$',
                r'$\mu_{I\to I,VSM}$',r'$\mu_{I\to E,all}$',r'$\mu_{I\to E,VSM}$']
pred_data[0] = pd.DataFrame(np.hstack((pred_sets[0],bal_sets[0],optr_sets[0],
                                       muX_sets[0],muE_sets[0],muI_sets[0])),columns=pred_labels)
pred_data[1] = pd.DataFrame(np.hstack((pred_sets[1],bal_sets[1],optr_sets[1],
                                       muX_sets[1],muE_sets[1],muI_sets[1])),columns=pred_labels)

mod_fam_dict = {}
mod_fam_dict['params'] = param_data
mod_fam_dict['param_labels'] = param_labels
mod_fam_dict['preds'] = pred_data
mod_fam_dict['pred_labels'] = pred_labels

output_dir='./'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_dir+"Struct-Model-Family"+'.pkl', 'wb') as handle:
    pickle.dump(mod_fam_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#!/usr/bin/python
import argparse

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd

import os
import time
try:
    import pickle5 as pickle
except:
    import pickle

import functions_optimal as fun
import data_analysis as da
import sims_utils as su
import validate_utils as vu
import plot_functions as pl

########################################################################################################################
########################################################################################################################
########################################################################################################################


# Define plotting style
font = {'family' : 'normal', 'weight' : 'normal', 'size' : 7, 'family' : 'serif', 'serif' : ['Arial']}
mpl.rc('font', **font)
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42


########################################################################################################################
# Output folder
output_dir='./perceptron_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

########################################################################################################################


########################################################################################################################
# .____                     .___ __________                    .__   __          
# |    |    _________     __| _/ \______   \ ____   ________ __|  |_/  |_  ______
# |    |   /  _ \__  \   / __ |   |       _// __ \ /  ___/  |  \  |\   __\/  ___/
# |    |__(  <_> ) __ \_/ /_/ |   |    |   \  ___/ \___ \|  |  /  |_|  |  \___ \ 
# |_______ \____(____  /\____ |   |____|_  /\___  >____  >____/|____/__| /____  >
#         \/         \/      \/          \/     \/     \/                     \/ 
#
########################################################################################################################

print(' ')
print('-----------------------------------------------------------------------------------------------')
print('-------------------------------------Loading results files-------------------------------------')
print('-----------------------------------------------------------------------------------------------')
print(' ')
parser = argparse.ArgumentParser(description=('This python script takes simulated optogenetic responses in networks'
    'that have been validated on multiple orientation maps, and trains a perceptron to predict a mapping '
    'between network parameters and response statistics to predict a best-fit model.'))

parser.add_argument('-nRep',                '--nRep',               help='number of least_squares fits done per seed', type=int, default=100)
parser.add_argument('-seedsim',             '--seedsim',            help='seed used for perceptron training and least_squares fitting',type=int, default=0)
parser.add_argument('-alpha',               '--alpha',              help='mlp alpha', type=float, default=0.00001)
parser.add_argument('-hiddenlayersizes',    '--hiddenlayersizes',   help='touple with the sizes of the hidden layers of the MLP regressor', nargs='+', type=int)
parser.add_argument('-learningrateinit',    '--learningrateinit',   help='initial learning rate of the MLP regressor', type=float, default=0.001)
parser.add_argument('-tol',                 '--tol',                help='tolerance of the MLP regressor', type=float, default=0.00001)
parser.add_argument('-beta1',               '--beta1',              help='beta 1 the MLP regressor', type=float, default=0.9)
parser.add_argument('-beta2',               '--beta2',              help='beta 2 MLP regressor', type=float, default=0.9999)



args = vars(parser.parse_args())
print(parser.parse_args())

nRep= args['nRep']
seedmlp= args['seedsim']

alpha=args['alpha']
hidden_layer_sizes=tuple(args['hiddenlayersizes'])
print(hidden_layer_sizes)
learning_rate_init=args['learningrateinit']
tol=args['tol']
print(tol)

beta_1=args['beta1']
beta_2=args['beta2']


print('You are running perceptron with args' + ' '.join([k+'='+str(v) for k,v in zip(args.keys(),args.values())]))

#####################################################
# Make data instance. This should always be both
this_animal='both'
monkey_mouse_data=da.Data_MonkeyMouse(this_animal,'../data')

Omap = 'separate'
tuned = 'yes'
RF = 'in'

start = time.process_time()
path2results='./../simulation_results/'
path2validate='./../validation_results/'
# _, jn =su.get_concatenated_results(path2results,'map',tuned,RF)
jn = 240
# jn is the last file number that exists, we are going to save the new simulations in there plus the seed number we send
print('your jn is :',jn)

param_sets = [None,None]
pred_sets = [None,None]

best_preds, best_params_fitted,residuals, loss = vu.get_best_preds_from_validate_param(path2validate,
                                                                        with_contrast=False,close='Both',cut=[5,15])
for anim_idx in range(len(monkey_mouse_data.this_animals)):
    param_sets[anim_idx] = np.array([item[list(vu.res_param_idxs.values())] for sublist in best_params_fitted[anim_idx]
                              for item in sublist])
    _,uniq_idxs = np.unique(param_sets[anim_idx],axis=0,return_index=True)
    param_sets[anim_idx] = param_sets[anim_idx][uniq_idxs,:]
    pred_sets[anim_idx] = np.array([item for sublist in best_preds[anim_idx] for item in sublist])
    pred_sets[anim_idx] = pred_sets[anim_idx][uniq_idxs]

param_setswc = [None,None]
pred_setswc = [None,None]

best_preds_w_c, best_params_fitted,best_rX,residuals_w_c, loss_w_c =\
    vu.get_best_preds_from_validate_param(path2validate, with_contrast=True,close='Both',cut=[5,15])
for anim_idx in range(len(monkey_mouse_data.this_animals)):
    param_sets_aux = np.array([item[list(vu.res_param_idxs.values())] for sublist in best_params_fitted[anim_idx]
                          for item in sublist])
    _,uniq_idxs = np.unique(param_sets_aux,axis=0,return_index=True)
    param_sets_aux = param_sets_aux[uniq_idxs,:]
    rX_sets_aux = np.array([item for sublist in best_rX[anim_idx] for item in sublist])
    rX_sets_aux = rX_sets_aux[uniq_idxs]
    pred_sets_aux = np.array([item for sublist in best_preds_w_c[anim_idx] for item in sublist])
    pred_sets_aux = pred_sets_aux[uniq_idxs]
    nc = rX_sets_aux.shape[1]
    
    param_setswc[anim_idx] = np.repeat(param_sets_aux,nc,axis=0)
    rX_sets_aux = np.reshape(rX_sets_aux,-1)
    param_setswc[anim_idx][:,su.sim_param_idxs['rX']] = rX_sets_aux
    
    pred_setswc[anim_idx] = np.reshape(np.transpose(pred_sets_aux,(0,2,1)),(-1,6))

params_dict_fixed={k:[0,500,0.09,30,1e-3,200][vu.res_param_idxs_fixed[k]] for k in list(vu.res_param_idxs_fixed.keys())}

anim_params = [None,None]
anim_preds = [None,None]

for anim_idx in range(len(monkey_mouse_data.this_animals)):
    anim_params[anim_idx] = np.vstack((param_sets[anim_idx],param_setswc[anim_idx]))
    anim_preds[anim_idx] = np.vstack((pred_sets[anim_idx],pred_setswc[anim_idx]))
    wide_spat_idxs = np.logical_and(anim_params[anim_idx][:,su.sim_param_idxs['SlE']] > 0.08,
                                    anim_params[anim_idx][:,su.sim_param_idxs['SlI']] > 0.08)
    anim_params[anim_idx] = anim_params[anim_idx][wide_spat_idxs,:]
    anim_preds[anim_idx] = anim_preds[anim_idx][wide_spat_idxs,:]
    _,uniq_idxs = np.unique(anim_params[anim_idx],axis=0,return_index=True)
    anim_params[anim_idx] = anim_params[anim_idx][uniq_idxs,:]
    anim_preds[anim_idx] = anim_preds[anim_idx][uniq_idxs,:]

best_preds_w_c, best_params_fitted,best_rX,residuals_w_c, loss_w_c =\
    vu.get_best_preds_from_validate_param(path2validate, with_contrast=True,close='Both',cut=[3,8],covcut=[1.2,4])
pl.plot_best_preds_from_validate_param(best_preds_w_c,with_contrast=True)
# print(best_params_fitted)

param_best_fit_setswc = [None,None]

for anim_idx in range(2):
    param_best_fit_sets_aux = np.array([item[list(vu.res_param_idxs.values())] for sublist in best_params_fitted[anim_idx]
                          for item in sublist])
    wide_spat_idxs = np.logical_and(param_best_fit_sets_aux[:,su.sim_param_idxs['SlE']] > 0.08,
                                    param_best_fit_sets_aux[:,su.sim_param_idxs['SlI']] > 0.08)
    param_best_fit_sets_aux = param_best_fit_sets_aux[wide_spat_idxs,:]
    _,uniq_idxs = np.unique(param_best_fit_sets_aux,axis=0,return_index=True)
    param_best_fit_setswc[anim_idx] = param_best_fit_sets_aux[uniq_idxs,:]

print('Mouse parameter results shape: ',anim_params[0].shape)
print('Monkey parameter results shape:',anim_params[1].shape)

print('')
print("Result loading took ",time.process_time() - start," s")



########################################################################################################################
# ___________             .__          _______          __                       __    
# \__    ___/___________  |__| ____    \      \   _____/  |___  _  _____________|  | __
#   |    |  \_  __ \__  \ |  |/    \   /   |   \_/ __ \   __\ \/ \/ /  _ \_  __ \  |/ /
#   |    |   |  | \// __ \|  |   |  \ /    |    \  ___/|  |  \     (  <_> )  | \/    < 
#   |____|   |__|  (____  /__|___|  / \____|__  /\___  >__|   \/\_/ \____/|__|  |__|_ \
#                       \/        \/          \/     \/                              \/
#
########################################################################################################################

print(' ')
print('----------------------------------------Training network---------------------------------------')
print(' ')

start = time.process_time()
nameout_sim='Omap='+str(Omap)+'_Tuned='+str(tuned)+'_RF='+str(RF)+'_seedmlp='+str(seedmlp)

predictor_sim = [None,None]
predictor_data = [None,None]
sim_predictor_data = [None,None]
sim_predictor_cost = [None,None]


################# YOU ARE HERE PUT THE PARAMS IN !!!

mplparams={}
mplparams['activation']='relu'
mplparams['alpha']=alpha
mplparams['hidden_layer_sizes']=hidden_layer_sizes
mplparams['learning_rate_init']=learning_rate_init
mplparams['tol']=tol
mplparams['beta_1']=beta_1
mplparams['beta_2']=beta_2


for anim_idx in range(len(monkey_mouse_data.this_animals)):
    predictor_sim[anim_idx],predictor_data[anim_idx] = fun.build_predictor_funs(anim_params[anim_idx],
        anim_preds[anim_idx],output_dir,monkey_mouse_data.this_animals[anim_idx]+'_'+nameout_sim,moments=False,mplparams=mplparams)
    sim_predictor_data[anim_idx],sim_predictor_cost[anim_idx] = fun.build_sim_predictor_funs(params_dict_fixed,
                                                                   'sp' if anim_idx == 0 else 'map')

print(' ')
print("Network training took ",time.process_time() - start," s")



########################################################################################################################
# ___________.__  __    ________          __          
# \_   _____/|__|/  |_  \______ \ _____ _/  |______   
#  |    __)  |  \   __\  |    |  \\__  \\   __\__  \  
#  |     \   |  ||  |    |    `   \/ __ \|  |  / __ \_
#  \___  /   |__||__|   /_______  (____  /__| (____  /
#      \/                       \/     \/          \/ 
#
########################################################################################################################

start = time.process_time()

nRep = 1
for fd in range(1):
    print(' ')
    print('----------------------------------Fitting Data Separately--------------------------------------')
    print(' ')
    # We set the random seed for the search
    np.random.seed(1000*seedmlp+fd)
    output_fit={}
    for anim_idx in range(len(monkey_mouse_data.this_animals)):
        dataset=monkey_mouse_data.bootstrap_moments[anim_idx]
        contrast=monkey_mouse_data.contrast[anim_idx]
        nc=monkey_mouse_data.nc[anim_idx]
        sol,cost=fun.fit_model_to_data(dataset,predictor_data[anim_idx],nc,nRep,monkey_mouse_data.this_animals[anim_idx],
                                          cost_model=sim_predictor_cost[anim_idx],sim_params=param_best_fit_setswc[anim_idx])

        anim_idx_best=np.argmin(cost)
        best_param_raw=sol[anim_idx_best,:]
        best_cost=cost[anim_idx_best]
        best_inputs=fun.fit_inputs_to_data_given_param(dataset,predictor_data[anim_idx],best_param_raw,
                                                          nc,monkey_mouse_data.this_animals[anim_idx])
        predictions=predictor_data[anim_idx](best_inputs,best_param_raw,nc,monkey_mouse_data.this_animals[anim_idx])
        best_sims_dict=su.get_params_dict_best_fit(best_param_raw,params_dict_fixed)

        output_fit['sol_'+monkey_mouse_data.this_animals[anim_idx]]=sol
        output_fit['cost_'+monkey_mouse_data.this_animals[anim_idx]]=cost
        output_fit['best_inputs_'+monkey_mouse_data.this_animals[anim_idx]]=best_inputs
        output_fit['best_param_raw_'+monkey_mouse_data.this_animals[anim_idx]]=best_param_raw
        output_fit['best_cost_'+monkey_mouse_data.this_animals[anim_idx]]=best_cost
        output_fit['predictions_'+monkey_mouse_data.this_animals[anim_idx]]=predictions
        output_fit['best_sims_dict_'+monkey_mouse_data.this_animals[anim_idx]]=best_sims_dict
        output_fit['nc']=monkey_mouse_data.nc

    nameout_fits='Model_Fit_Separately_'+'StructModel'+'-seednnls='+str(seedmlp)+'-'+ '-'.join([k+'='+str(v) for k,v in zip(mplparams.keys(),mplparams.values())])
    nameout_fits_path=output_dir+'/'+ nameout_fits
    print('Saving all fit stuff in'+  nameout_fits_path)

    with open(nameout_fits_path+".pkl", 'wb') as handle_Model:
        pickle.dump(output_fit, handle_Model, protocol=pickle.HIGHEST_PROTOCOL)
    print(' ')
    print("Separate fits took ",time.process_time() - start," s")
       
    print(' ')
    print('----------------------------------Ploting the Spearate fit-------------------------------------')
    print(' ')
    pl.plot_fitted_data(output_dir,nameout_fits)


    ##############################################################################################
    #__________                     .__              .__          __  .__
    #\______   \__ __  ____     _____|__| _____  __ __|  | _____ _/  |_|__| ____   ____   ______
    # |       _/  |  \/    \   /  ___/  |/     \|  |  \  | \__  \\   __\  |/  _ \ /    \ /  ___/
    # |    |   \  |  /   |  \  \___ \|  |  Y Y  \  |  /  |__/ __ \|  | |  (  <_> )   |  \\___ \
    # |____|_  /____/|___|  / /____  >__|__|_|  /____/|____(____  /__| |__|\____/|___|  /____  >
    #        \/           \/       \/         \/                \/                    \/     \/
    ###############################################################################################




    print(' ')
    print('-----------------------------------Simulating Fit Params---------------------------------------')
    print(' ')

    start = time.process_time()
    this_jn=jn+1000*seedmlp+fd

    su.get_simulations_from_best_fit(output_dir, nameout_fits,stim_size=0.5, this_animals=['mouse','monkey'],
        resultsdir=path2results, this_jn=this_jn)
    print(' ')
    print("Simulating fit params took ",time.process_time() - start," s")

    ##################### HERE WE NEED TO MODIFY pl.plot_fitted_data(nameout) to include the all_moments
       
    print(' ')
    print('-----------------------------Ploting the Spearate fit with Sims--------------------------------')
    print(' ')
    pl.plot_fitted_data(output_dir,nameout_fits)

print('Done')

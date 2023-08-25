#!/usr/bin/python
import argparse

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize, least_squares

import os
import time
import pickle

import functions_optimal as funopt
import functions as fun
import data_analysis as da
import sims_utils as su
import validate_utils as vu
import plot_functions as pl
import ricciardi_class as ric
import network as network




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
# Which_results
parser = argparse.ArgumentParser(description=('This python script takes results from sampled spatial model parameters, '
    'trains a net to interpolate the results, and finds parameters that best fit the experimental results'))


parser.add_argument('-animal',   '--animal',    help='animal',type=str, default='monkey')
parser.add_argument('-paramidx', '--paramidx',  help=' index', type=int, default=0)
parser.add_argument('-inputic',  '--inputic',   help='index of contrast',type=int,default=0)

args = vars(parser.parse_args())

name_end='-'.join([kk+'_'+str(ll) for kk,ll in args.items()])


animal = args['animal']
paramidx= args['paramidx']
inputic= args['inputic']


############## I changed the name of the files here, for that I used (in terminal)
#>> brew install mmv
#>> mmv '*results*' '#1results_Omap=map_Tuned=yes_RF=all#2'


#--------------------------------------------------------------------------

ri = ric.Ricciardi()
ri.set_up_nonlinearity()

#--------------------------------------------------------------------------
# Output folder

resultsdir = "./../results_new_format_with_GI/"
validatedir = "./../validate_with_GI/"


if not os.path.exists(validatedir):
    os.makedirs(validatedir)

########################################################################################################################
# .____                     .___ __________                    .__   __
# |    |    _________     __| _/ \______   \ ____   ________ __|  |_/  |_  ______
# |    |   /  _ \__  \   / __ |   |       _// __ \ /  ___/  |  \  |\   __\/  ___/
# |    |__(  <_> ) __ \_/ /_/ |   |    |   \  ___/ \___ \|  |  /  |_|  |  \___ \
# |_______ \____(____  /\____ |   |____|_  /\___  >____  >____/|____/__| /____  >
#         \/         \/      \/          \/     \/     \/                     \/
#
########################################################################################################################
#--------------------------------------------------------------------------
# Load results
tuned = 'yes'
rf = 'in'
filter='none'

start = time.process_time()

results_map_aux, jn_map = su.get_concatenated_results(resultsdir,'map',tuned,rf)
results_sp_aux, jn_sp = su.get_concatenated_results(resultsdir,'sp',tuned,rf)
    

results_map=su.return_filtered_results(results_map_aux,filter)
results_sp=su.return_filtered_results(results_sp_aux,filter)

params_map_fixed = results_map[0,list(su.res_param_idxs_fixed.values())]
sim_map_params = results_map[:,list(su.res_param_idxs.values())]
sim_map_moments = results_map[:,list(su.res_moment_idxs.values())]

params_sp_fixed = results_sp[0,list(su.res_param_idxs_fixed.values())]
sim_sp_params = results_sp[:,list(su.res_param_idxs.values())]
sim_sp_moments = results_sp[:,list(su.res_moment_idxs.values())]

#--------------------------------------------------------------------------
# Load data (the data is the mean of a bootstrap thats why we set the random seed)

np.random.seed(0)
data=da.Data_MonkeyMouse('both','../data')

dataset = data.bootstrap_moments
contrast = data.contrast

#--------------------------------------------------------------------------
# Parameters of the search

find_by_cut=False
find_by_sorted=True
seeds = [1,3,5,7]#np.round(np.random.rand(4)*2**32)
cut = 3
cut_sim_loss=3.5
max_min = 10
nrX = 20

#--------------------------------------------------------------------------
# Initialize and find order of ranked params



res={}
loss={}


if animal =='mouse':
    animal_idx=0
    this_idx=vu.untuned_idxs
    results=results_sp
    ori_type='saltandpepper'
    nc = data.nc[animal_idx]
    number_of_params=np.ones(nc,dtype=int)
    rXs = np.linspace(1,50,nrX)#3*10**(np.linspace(-0.1,0.9,nrX)*2-1)
    
elif animal =='monkey':
    animal_idx=1
    this_idx=vu.tuned_idxs
    results=results_map
    ori_type='columnar'
    nc = data.nc[animal_idx]
    number_of_params=np.ones(nc,dtype=int)
    number_of_params[-1]=10
    rXs = np.linspace(1,50,nrX)#5*10**(np.linspace(-0.1,0.9,nrX)*2-1)


# Get parameters fixed-------------------------------------
params_fixed = results[0,list(su.res_param_idxs_fixed.values())]
KX = params_fixed[1]
pmax = params_fixed[2]
SoriE = params_fixed[3]
Lam = params_fixed[4]
Tmax_over_tau_E = params_fixed[5]
T = np.arange(0,1.5*Tmax_over_tau_E*ri.tau_E,ri.tau_I/3);
mask_time=T>(0.5*Tmax_over_tau_E*ri.tau_E)

# Get Sim params--------------------------------------------
params = results[:,list(su.res_param_idxs.values())]

# Get moments
moments = results[:,list(su.res_moment_idxs.values())]

# Initialize -----------------------------------------------
res=np.zeros((results.shape[0],6,nc))
loss=np.zeros((results.shape[0],nc))
this_sim_preds = np.zeros((results.shape[0],6))

# Find the loss---------------------------------------------
this_sim_preds[:,0] = moments[:,this_idx['mean_base']]
this_sim_preds[:,1] = moments[:,this_idx['mean_base']] + moments[:,this_idx['mean_delta']]
this_sim_preds[:,2] = moments[:,this_idx['std_base']]
this_sim_preds[:,3] = np.sqrt(moments[:,this_idx['std_base']]**2 +\
                                 moments[:,this_idx['std_delta']]**2 +\
                                 2*moments[:,this_idx['cov']])
this_sim_preds[:,4] = moments[:,this_idx['std_delta']]
this_sim_preds[:,5] = moments[:,this_idx['cov']] / moments[:,this_idx['std_delta']]**2

for i in range(nc):
    contrast_dataset = data.bootstrap_moments[animal_idx][:,i,:]
    res[:,:,i] = (this_sim_preds-contrast_dataset[:,0])/contrast_dataset[:,1]
    loss[:,i] = 0.5*np.sum(res[:,:,i]**2,1)
    
#--------------------------------------------------------------------------
# Starting validation

start = time.process_time()
print('VALIDATING CONTRAST *' + str(inputic+1) + '* of the ' + str(animal)+' **** ')
contrast_dataset = dataset[animal_idx][:,inputic,:]

# Running the paramidx-th of the best fitted params------------------------

this_idx_param_sorted=np.argsort(loss[:,inputic])
loss_sorted=loss[this_idx_param_sorted,inputic]
best_params = params[this_idx_param_sorted,:]
this_params= best_params[paramidx]
print(" The loss of the current parameter set is " +str(loss_sorted[paramidx]))

preds= np.zeros((len(seeds),6))
seed_idx = 0

J = this_params[su.sim_param_idxs['J']]
gE = this_params[su.sim_param_idxs['gE']]
gI = this_params[su.sim_param_idxs['gI']]
beta = this_params[su.sim_param_idxs['beta']]
rX = this_params[su.sim_param_idxs['rX']]

CV_K = this_params[su.sim_param_idxs['CV_K']]
SlE = this_params[su.sim_param_idxs['SlE']]
SlI = this_params[su.sim_param_idxs['SlI']]
SoriI = this_params[su.sim_param_idxs['SoriI']]
Stun = this_params[su.sim_param_idxs['Stun']]
CV_Lam = this_params[su.sim_param_idxs['CV_Lam']]
L = this_params[su.sim_param_idxs['L']]
GI = this_params[su.sim_param_idxs['GI']]

for seed_idx,seed_con in enumerate(seeds):
    print(' - seed ' + str(seed_idx+1) + ' out of ' + str(len(seeds)))

    net = network.network(seed_con=int(seed_con), n=2, Nl=25, NE=8, gamma=0.25, dl=1,
                          Sl=np.array([[SlE,SlI],[SlE,SlI]]), Sori=np.array([[SoriE,SoriI],[SoriE,SoriI]]),
                          Stun=Stun, ori_type=ori_type)

    net.GI = GI

    net.generate_disorder(J,gE,gI,beta,pmax,CV_K,rX,KX,Lam,CV_Lam,0.5,vanilla_or_not=False)
    this_moments,_,_ = fun.get_moments_of_r_sim(net,ri,T,mask_time,L,rf,tuned,False,max_min)
    
    preds[seed_idx,0] = this_moments[this_idx['mean_base']]
    preds[seed_idx,1] = this_moments[this_idx['mean_base']] + this_moments[this_idx['mean_delta']]
    preds[seed_idx,2] = this_moments[this_idx['std_base']]
    preds[seed_idx,3] = np.sqrt(this_moments[this_idx['std_base']]**2 +\
                                     this_moments[this_idx['std_delta']]**2 +\
                                     2*this_moments[this_idx['cov']])
    preds[seed_idx,4] = this_moments[this_idx['std_delta']]
    preds[seed_idx,5] = this_moments[this_idx['cov']] / this_moments[this_idx['std_delta']]**2


mask = np.invert(np.any(preds > 1e4,1))
try:
    mean_preds = np.mean(preds[mask,:],0)
except:
    mean_preds = np.zero(6)
mean_preds[np.isnan(mean_preds)==True]=0

this_res = (mean_preds-contrast_dataset[:,0])/contrast_dataset[:,1]
#if animal =='monkey' and inputic==nc-1:
#    this_res = (mean_preds-contrast_dataset[:,0])*1.5/contrast_dataset[:,1]
#    print('      More weight on neg cov')
this_loss = 0.5*np.sum(this_res**2)
print('The actual loss of this paramset is ' + str(this_loss))
print('')
print("Validating predictions took ",time.process_time() - start," s")
print('')

#--------------------------------------------------------------------------
# SAVE output 
start = time.process_time()
name_out_animal_loss=validatedir+'Param_'+name_end

results=np.zeros((len(params_fixed)
                  +len(this_params)
                 +len(mean_preds)
                 +len(this_res)+1))

results[0:len(params_fixed)]=params_fixed
results[len(params_fixed):(len(params_fixed)+len(this_params))]=this_params
results[(len(params_fixed)+len(this_params)):(len(params_fixed)+len(this_params)+len(mean_preds))]=\
    mean_preds
results[(len(params_fixed)+len(this_params)+len(mean_preds)):-1]=this_res
results[-1]=this_loss
f_handle = open(name_out_animal_loss,'w')
np.savetxt(f_handle,results,fmt='%.6f', delimiter='\t')
f_handle.close()
print('')
print("Saving validated parameter results took ",time.process_time() - start," s")
print('')

if this_loss < cut_sim_loss:
    print('Accepting parameters')
    print('')
    print('Finding mapping between rX and contrast')
    
    start = time.process_time()
    rX_preds = np.zeros((len(rXs),6))
    for rX_idx,this_rX in enumerate(rXs):
        start = time.process_time()
        print(' - rX number ' + str(rX_idx+1) + ' out of ' + str(len(rXs)))
        this_rX_preds = np.zeros((len(seeds),6))
        for seed_idx,seed_con in enumerate(seeds):
            print('    - seed ' + str(seed_idx+1) + ' out of ' + str(len(seeds)))

            net = network.network(seed_con=int(seed_con), n=2, Nl=25, NE=8, gamma=0.25, dl=1,
                                  Sl=np.array([[SlE,SlI],[SlE,SlI]]), Sori=np.array([[SoriE,SoriI],[SoriE,SoriI]]),
                                  Stun=Stun, ori_type=ori_type)

            net.GI = GI

            net.generate_disorder(J,gE,gI,beta,pmax,CV_K,this_rX,KX,Lam,CV_Lam,0.5,vanilla_or_not=False)
            this_moments,_,_ = fun.get_moments_of_r_sim(net,ri,T,mask_time,L,rf,tuned,False,max_min)
            this_rX_preds[seed_idx,0] = this_moments[this_idx['mean_base']]
            this_rX_preds[seed_idx,1] = this_moments[this_idx['mean_base']] + this_moments[this_idx['mean_delta']]
            this_rX_preds[seed_idx,2] = this_moments[this_idx['std_base']]
            this_rX_preds[seed_idx,3] = np.sqrt(this_moments[this_idx['std_base']]**2 +\
                                             this_moments[this_idx['std_delta']]**2 +\
                                             2*this_moments[this_idx['cov']])
            this_rX_preds[seed_idx,4] = this_moments[this_idx['std_delta']]
            this_rX_preds[seed_idx,5] = this_moments[this_idx['cov']] / this_moments[this_idx['std_delta']]**2


        this_rX_mask = np.invert(np.any(this_rX_preds > 1e4,1))
        try:
            mean_rX_preds = np.mean(this_rX_preds[this_rX_mask,:],0)
        except:
            mean_rX_preds = np.zero(6)
        mean_rX_preds[np.isnan(mean_rX_preds)==True]=1e4+this_rX

        rX_preds[rX_idx,:] = mean_rX_preds

        if np.count_nonzero(this_rX_mask) == 0: # timeout for all seeds
            rX_preds[rX_idx+1:,:] = 1e4+rXs[rX_idx+1:]
            break
    
    print('')
    print("Calculating predicitons at different rX took ",time.process_time() - start," s")
    print('')

    #--------------------------------------------------------------------------
    # R_X interpolation needs to be done

    start = time.process_time()
    # interpolate rX vs predictions
    mean_base_itp = interp1d(rXs, rX_preds[:,0], kind='linear', fill_value='extrapolate')
    mean_opto_itp = interp1d(rXs, rX_preds[:,1], kind='linear', fill_value='extrapolate')
    std_base_itp  = interp1d(rXs, rX_preds[:,2], kind='linear', fill_value='extrapolate')
    std_opto_itp  = interp1d(rXs, rX_preds[:,3], kind='linear', fill_value='extrapolate')
    std_delta_itp = interp1d(rXs, rX_preds[:,4], kind='linear', fill_value='extrapolate')
    norm_cov_itp  = interp1d(rXs, rX_preds[:,5], kind='linear', fill_value='extrapolate')

    # define function to map set of rX to predictions
    def prediction_data(inputs):
        prediction=np.zeros((6,nc))
        prediction[0,:] = mean_base_itp(inputs)
        prediction[1,:] = mean_opto_itp(inputs)
        prediction[2,:] = std_base_itp(inputs)
        prediction[3,:] = std_opto_itp(inputs)
        prediction[4,:] = std_delta_itp(inputs)
        prediction[5,:] = norm_cov_itp(inputs)
        return prediction

    # define function to map set of rX to residuals from data
    def Residuals_raw(inputs):
        prediction=prediction_data(inputs)
        prediction[np.isnan(prediction)==True]=1e10
        res=(prediction-dataset[animal_idx][:,:,0])/dataset[animal_idx][:,:,1]
        return res

    def Residuals(inputs):
        res = Residuals_raw(inputs)
        return res.ravel()

    input_0=np.linspace(rXs[0]+1,rXs[-1]/2-1,nc)

    #print(input_0)
    res_2 = least_squares(Residuals, input_0,
                      bounds=(rXs[0]-1, rXs[-1]+5))
    inputs=res_2.x
    fit_rX = inputs
    fit_preds = prediction_data(inputs)
    fit_res = Residuals_raw(inputs)
    fit_loss = 0.5*np.sum(fit_res**2,0)
    
    print('')
    print("Calculating predicitons at different rX took ",time.process_time() - start," s")
    print('')

    print('Accepting parameters and saving validated set ')


    #--------------------------------------------------------------------------
    # SAVE output 

    name_out_animal_loss_wc=validatedir+'Param_w_contrast_'+name_end

    rX_results=np.zeros((len(results)
                     +len(fit_rX)
                     +len(fit_preds.ravel())
                     +len(fit_res.ravel())
                     +len(fit_loss)))


    rX_results[0:len(results)]=results
    rX_results[len(results):(len(results)+len(fit_rX))]=fit_rX
    rX_results[(len(results)+len(fit_rX)):(len(results)+len(fit_rX)+len(fit_preds.ravel()))]=\
        fit_preds.ravel()
    rX_results[(len(results)+len(fit_rX)+len(fit_preds.ravel())):(len(results)+len(fit_rX)+len(fit_preds.ravel())+\
        len(fit_res.ravel()))]=fit_res.ravel()
    rX_results[(len(results)+len(fit_rX)+len(fit_preds.ravel())+len(fit_res.ravel()))::]=fit_loss
    this_f_handle = open(name_out_animal_loss_wc,'w')
    np.savetxt(this_f_handle,rX_results,fmt='%.6f', delimiter='\t')
    this_f_handle.close()
    print('')
    print("Saving validated parameter with contrast results took ",time.process_time() - start," s")
    print('')

    #--------------------------------------------------------------------------
    # maybe here calling sims close to best param?



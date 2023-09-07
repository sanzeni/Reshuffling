#!/usr/bin/python
import sys

import argparse
#%load_ext autoreload
#%autoreload 2
#%matplotlib inline 
import matplotlib.pyplot as plt

import numpy as np
import matplotlib as mpl
from scipy import stats

import functions_optimal as fun


import pandas as pd


__author__ = 'sandro'
parser = argparse.ArgumentParser(description='This is a demo script by Sandro.')
parser.add_argument('-p','--params',help='Filename for results', required=True)
parser.add_argument('-r','--results',help='Filename for results', required=True)
parser.add_argument('-J', '--jobnumber', help='job number', required=False)
args = parser.parse_args()


print('Loading params file')
pdf = pd.read_csv('simulation_param_Fixed_J_and_CVopto.txt',delim_whitespace=True) #header=None
print(args)
jn = int(args.jobnumber)



idx_species=int(pdf.idx_species.iloc[jn]);
'''
g_E=
g_I=
beta=
'''
log10_CVopto=pdf.log10_CVopto.iloc[jn];
log10_J=pdf.log10_J.iloc[jn];
nRep=int(pdf.nRep.iloc[jn]);


print('Parameters used')
print(idx_species,log10_CVopto,log10_J,nRep)

print('fit network  to simulation')

param_min=[ 3.  ,  2. ,  -1. ,  -3.52, -1. ,  -5.  ]
param_max=[10.  ,  9.49,  1. ,  -0.52 , 1. ,  -2.3 ]
Predictor_sim,Predictor_data=fun.build_function()

print('Create dataset')
data_mice=np.loadtxt('Mice_with_trials.txt');
data_monkeys=np.loadtxt('Monkeys_with_trials.txt');
data_both_species=[data_mice,data_monkeys]

dataset,Con,nCon=fun.build_dataset(data_both_species[idx_species])


print('fit simulations to data')
sol,cost=fun.fit_model_to_data_fixed_CVopto_and_log10_J(log10_CVopto,log10_J,dataset,Predictor_data,nCon,nRep,param_min,param_max)

idx_best=np.argmin(cost)
best_param=sol[idx_best,:]
best_cost=cost[idx_best]
best_param=np.concatenate((best_param,[log10_CVopto,log10_J]))
best_inputs=fun.fit_inputs_to_data_given_param(dataset,Predictor_data,best_param,nCon)

print(best_param,best_inputs,best_cost)

print('Saving results')
# simulations param+mean results+ meaurements of rate convergence

results=np.zeros((1,len(best_param)
                  +len(best_inputs)
                 +1))

results[0,0:len(best_param)]=best_param[:]
results[0,len(best_param):(len(best_param)+len(best_inputs))]=best_inputs[:]
results[0,(len(best_param)+len(best_inputs))]=best_cost

# Clean file to print results
f_handle = open('results/'+args.results,'w')
np.savetxt(f_handle,results,fmt='%.6f', delimiter='\t')
f_handle.close()

print('Done')










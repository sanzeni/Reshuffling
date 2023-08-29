#!/usr/bin/python
import sys
import os
import argparse
import functions as fun

import numpy as np
import gc

from scipy.interpolate import interp1d

import pandas as pd
import ricciardi_class as ricciardi_class
import network as network
import functions as fun

import time

ri=ricciardi_class.Ricciardi()
import sims_utils as su
########################################################################################################################
#.____                     .___
#|    |    _________     __| _/ ___________ ____________    _____   ______
#|    |   /  _ \__  \   / __ |  \____ \__  \\_  __ \__  \  /     \ /  ___/
#|    |__(  <_> ) __ \_/ /_/ |  |  |_> > __ \|  | \// __ \|  Y Y  \\___ \
#|_______ \____(____  /\____ |  |   __(____  /__|  (____  /__|_|  /____  >
#        \/         \/      \/  |__|       \/           \/      \/     \/
#
########################################################################################################################

parser = argparse.ArgumentParser(description=('This python script samples parameters for an unstructured model, '
    'simulates the network, and saves the statistics of the network.'))
parser.add_argument('-J',    '--jobnumber',help='job number',type=int,required=True)

args = vars(parser.parse_args())



name_end='structless'

# read common simulation params
print(' ')
print('----------------------------------------------------------------------------------------------------')
print('-------------------------------------Loading params file--------------------------------------------')
print('----------------------------------------------------------------------------------------------------')
print(' ')
pdf = pd.read_csv('./../simulation_param.txt',delim_whitespace=True) #header=None
print(parser.parse_args())
jn = int(args['jobnumber'])

######### Read some  params
sigma_t=0.01
tau_rp=0.002
K=int(pdf.K.iloc[jn]);
Tmax_over_tau_E=int(pdf.Tmax_over_tau_E.iloc[jn]);
seed=int(pdf.seed.iloc[jn])
nrep=int(pdf.nrep.iloc[jn]);

T=np.arange(0,1.5*Tmax_over_tau_E*ri.tau_E,ri.tau_I/3);
mask_time=T>(0.5*Tmax_over_tau_E*ri.tau_E)

######### Set up Ricciardi
ri.set_up_nonlinearity()


resultsdir='./simulation_results/'
name_results='results_'+name_end+'_'+str(jn)+'.txt'
this_results=resultsdir+name_results
print('Saving all results in '+  name_results)
print(' ')

init = True
try:
    this_loaded_results=np.loadtxt(this_results)
    first_rep = this_loaded_results.shape[0]
except:
    first_rep = 0

# if not os.path.exists(this_results_folder):
#     os.makedirs(this_results_folder)

########################################################################################################################
#   _____          __                           __                       __
#  /     \ _____  |  | __ ____     ____   _____/  |___  _  _____________|  | __
# /  \ /  \\__  \ |  |/ // __ \   /    \_/ __ \   __\ \/ \/ /  _ \_  __ \  |/ /
#/    Y    \/ __ \|    <\  ___/  |   |  \  ___/|  |  \     (  <_> )  | \/    <
#\____|__  (____  /__|_ \\___  > |___|  /\___  >__|   \/\_/ \____/|__|  |__|_ \
#        \/     \/     \/    \/       \/     \/                              \/
#
########################################################################################################################

"""
Define core parameters for the network

Parameters
----------
stim_size : float
    Fraction of grid that visual RF covers
p : float
    Maximum connection probabiility
Lam : float
    mean opsin expression (V)
"""
stim_size=50.0;
p=0.1;
Lam=1*10**-3;

# sample parameters, simulate network, and save the statistics of the rates for each repetition
for idx_rep in range(first_rep,nrep):
    start = time.process_time()
    print('-----------------'+ 'Computing and saving network response for repetition ' + str(idx_rep) + ' out of '+str(nrep)+ '-----------------')


    
    J,GI,gE,gI,beta,CV_K,rX,L,CV_Lam=su.get_random_model_variable(['J','gE','gI','beta',\
                            'CV_K','rX','L','CV_Lam'])


    seed_con = (seed+idx_rep)%2**32

    ori_type = 'saltandpepper'
    net=network.network(seed_con=seed_con, n=2, Nl=25, NE=8, gamma=0.25, dl=1, ori_type=ori_type)

    print("Parameters used seed= {:d} // gE= {:.2f} // gI= {:.2f} // beta= {:.2f} // K= {:d} // p= {:.2f} // CV_K= {:.4f}" \
    .format(seed_con,gE,gI,beta,K,p,CV_K))
    print("Lam= {:.3f} // CV_Lam= {:.2f} // J= {:.6f} // rX= {:.2f} // L= {:.2f} // Tmax_over_tau_E= {:d}"\
    .format(Lam,CV_Lam,J,rX,L,Tmax_over_tau_E))
    print('')

    #------------------------------------------------------------------------------------------------------
    # generate network weight matrix, visual input matrix, and optogenetic input matrix
    #------------------------------------------------------------------------------------------------------

    net.generate_disorder(J,gE,gI,beta,p,CV_K,rX,K,Lam,CV_Lam,stim_size,vanilla_or_not=True)
    print("Disorder generation took ",time.process_time() - start," s")
    print('')
    start = time.process_time()
    if idx_rep==0:
        moments_of_r_sim, r_convergence, r_pippo,RATES,_,_,_ = fun.get_moments_of_r_sim(net,ri,T,mask_time,L,True)
    else:
        moments_of_r_sim, r_convergence, r_pippo = fun.get_moments_of_r_sim(net,ri,T,mask_time,L,False)
    print('')
    print("ODE integration took ",time.process_time() - start," s")
    print('')
    print('--------------------------------------------Saving results------------------------------------------')
    print(' ')



    
    #------------------------------------------------------------------------------------------------------
    # simulations param + mean results + meaurements of rate convergence
    #------------------------------------------------------------------------------------------------------
    start = time.process_time()
    sim_param=np.asarray([sigma_t,tau_rp,gE,gI,beta,K,p,CV_K,1,
          Lam,CV_Lam,J,rX,L,Tmax_over_tau_E])
    sim_results=moments_of_r_sim
    sim_convergence=r_convergence

    additional_measurements=r_pippo
    if init:
        results=np.zeros((nrep,len(sim_param)
                          +len(sim_results)
                         +len(sim_convergence)
                         +len(additional_measurements)))
        try:
            results[:first_rep,:] = np.loadtxt(this_results)
        except:
            pass
        init = False

    results[idx_rep,0:len(sim_param)]=sim_param[:]
    results[idx_rep,len(sim_param):(len(sim_param)+len(sim_results))]=sim_results
    results[idx_rep,(len(sim_param)+len(sim_results)):(len(sim_param)+len(sim_results)+len(sim_convergence))]=\
        sim_convergence
    results[idx_rep,(len(sim_param)+len(sim_results)+len(sim_convergence))::]=additional_measurements

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
    
    gc.collect()

print('Done')









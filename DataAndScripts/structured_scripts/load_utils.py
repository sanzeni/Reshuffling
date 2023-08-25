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
import inspect

from scipy.interpolate import interp1d

import pandas as pd
import network as network
import functions as fun
import functions as funsim
import time
import sims_utils as su
import validate_utils as vu
import data_analysis as da
#import ricciardi_class as ricciardi_class
#ri=ricciardi_class.Ricciardi()

data=da.Data_MonkeyMouse('both','../data')

def sim_avg_map(params_dict,rX_vec,ri,T,mask_time,RF,tuned,seeds,max_min,return_inputs=False,return_rates=False,input_var=False):
    this_params_dict = params_dict.copy()
    this_params_dict['input_var'] = input_var
    this_params_dict['Sl'] = np.array([[params_dict['SlE'],params_dict['SlI']],[params_dict['SlE'],params_dict['SlI']]])
    this_params_dict['Sori'] = np.array([[params_dict['SoriE'],params_dict['SoriI']],[params_dict['SoriE'],params_dict['SoriI']]])

    moments = np.zeros((len(rX_vec),10))
    preds_tuned = np.zeros((len(rX_vec),6))
    preds_untuned = np.zeros((len(rX_vec),6))
    bals = np.zeros((len(rX_vec),6))
    optrs = np.zeros((len(rX_vec),6))
    if return_inputs:
        muXs = np.zeros((len(rX_vec),6))
        muEs = np.zeros((len(rX_vec),6))
        muIs = np.zeros((len(rX_vec),6))
    if return_rates:
        rates = {}

    for rX_idx,this_rX in enumerate(rX_vec):
        print('Doing contrast '+str(rX_idx+1) +' of '+str(len(rX_vec)))

        this_params_dict['rX'] = this_rX

        rX_moments = np.zeros((len(seeds),10))
        rX_preds_tuned = np.zeros((len(seeds),6))
        rX_preds_untuned = np.zeros((len(seeds),6))
        rX_bals = np.zeros((len(seeds),6))
        rX_optrs = np.zeros((len(seeds),6))
        if return_inputs:
            rX_muXs = np.zeros((len(seeds),6))
            rX_muEs = np.zeros((len(seeds),6))
            rX_muIs = np.zeros((len(seeds),6))
        if return_rates:
            rX_rates = np.zeros((len(seeds),2,6250))

        for seed_idx,seed in enumerate(seeds):
            this_params_dict['seed_con'] = int(seed)
            filtered_mydict_net = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
                                                            inspect.signature(network.network).parameters.values()]}
            net = network.network(**filtered_mydict_net)
            net.GI = this_params_dict['GI']
            if 'GE' in this_params_dict:
                net.GE = this_params_dict['GE']
            filtered_mydict_disorder = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
                                                inspect.signature(net.generate_disorder).parameters.values()]}
            net.generate_disorder(**filtered_mydict_disorder)

            ori_idx=net.get_oriented_neurons(delta_ori=22.5)
            rf_idx=net.get_centered_neurons(stim_size=0.6)
            tune_idx=np.intersect1d(ori_idx,rf_idx)
            tuneE_idx=np.intersect1d(tune_idx,net.allE)
            tuneI_idx=np.intersect1d(tune_idx,net.allI)

            this_moments, _, _, this_rates, this_muEs, this_muIs, this_muLs = \
                fun.get_moments_of_r_sim(net,ri,T,mask_time,params_dict['L'],RF,tuned,return_activity=True,
                    max_min=max_min,return_dynas=False)
            if return_inputs:
                this_muXs = net.H
                this_muXs[net.allE] = ri.tau_E*this_muXs[net.allE]
                this_muXs[net.allI] = ri.tau_I*this_muXs[net.allI]
                this_muErecs = this_muEs - this_muXs

            rX_moments[seed_idx,:] = this_moments
            rX_preds_tuned[seed_idx,:] = vu.get_predictions_from_moments_and_idxs(this_moments,vu.tuned_idxs)
            rX_preds_untuned[seed_idx,:] = vu.get_predictions_from_moments_and_idxs(this_moments,vu.untuned_idxs)
            for i in range(6):
                if i == 0:
                    idxs = np.arange(net.N)
                elif i == 1:
                    idxs = tune_idx
                elif i == 2:
                    idxs = net.allE
                elif i == 3:
                    idxs = tuneE_idx
                elif i == 4:
                    idxs = net.allI
                elif i == 5:
                    idxs = tuneI_idx
                rX_bals[seed_idx,i] = np.mean(np.abs(this_muEs[idxs]+this_muIs[idxs])/this_muEs[idxs])
                rX_optrs[seed_idx,i] = np.mean(this_muLs[idxs]/this_muEs[idxs])
                if return_inputs:
                    rX_muXs[seed_idx,i] = np.mean(this_muXs[idxs])
                    rX_muEs[seed_idx,i] = np.mean(this_muErecs[idxs])
                    rX_muIs[seed_idx,i] = np.mean(this_muIs[idxs])
            if return_rates:
                rX_rates[seed_idx,:,:] = this_rates
        
        mask = np.invert(np.any(rX_moments > 1e4,1))
        try:
            moments[rX_idx,:] = np.mean(rX_moments[mask,:],0)
            preds_tuned[rX_idx,:] = np.mean(rX_preds_tuned[mask,:],0)
            preds_untuned[rX_idx,:] = np.mean(rX_preds_untuned[mask,:],0)
            bals[rX_idx,:] = np.mean(rX_bals[mask,:],0)
            optrs[rX_idx,:] = np.mean(rX_optrs[mask,:],0)
            if return_inputs:
                muXs[rX_idx,:] = np.mean(rX_muXs[mask,:],0)
                muEs[rX_idx,:] = np.mean(rX_muEs[mask,:],0)
                muIs[rX_idx,:] = np.mean(rX_muIs[mask,:],0)
            if return_rates:
                rates[rX_idx] = rX_rates #np.hstack([rX_rates[i,:,:] for i in np.arange(len(seeds))[mask]])
        except:
            moments[rX_idx,:] = 1e6*np.ones(10)
            preds_tuned[rX_idx,:] = 1e6*np.ones(6)
            preds_untuned[rX_idx,:] = 1e6*np.ones(6)
            bals[rX_idx,:] = 1e6*np.ones(6)
            optrs[rX_idx,:] = 1e6*np.ones(6)
            if return_inputs:
                muXs[rX_idx,:] = 1e6*np.ones(6)
                muEs[rX_idx,:] = 1e6*np.ones(6)
                muIs[rX_idx,:] = 1e6*np.ones(6)
            if return_rates:
                rates[rX_idx] = 1e6*np.ones((2,6250))

        moments[rX_idx,np.isnan(moments[rX_idx,:])==True] = 1e6
        preds_tuned[rX_idx,np.isnan(preds_tuned[rX_idx,:])==True] = 1e6
        preds_untuned[rX_idx,np.isnan(preds_untuned[rX_idx,:])==True] = 1e6
        bals[rX_idx,np.isnan(bals[rX_idx,:])==True] = 1e6
        optrs[rX_idx,np.isnan(optrs[rX_idx,:])==True] = 1e6
        if return_inputs:
            muXs[rX_idx,np.isnan(muXs[rX_idx,:])==True] = 1e6
            muEs[rX_idx,np.isnan(muEs[rX_idx,:])==True] = 1e6
            muIs[rX_idx,np.isnan(muIs[rX_idx,:])==True] = 1e6
#        if return_rates:
#            rates[rX_idx][np.isnan(rates[rX_idx])==True] = 1e6

    if return_inputs and not return_rates:
        return moments,preds_tuned,preds_untuned,bals,optrs,muXs,muEs,muIs
    elif return_rates and not return_inputs:
        return moments,preds_tuned,preds_untuned,bals,optrs,rates
    elif return_rates and return_inputs:
        return moments,preds_tuned,preds_untuned,bals,optrs,muXs,muEs,muIs,rates
    else:
        return moments,preds_tuned,preds_untuned,bals,optrs


def sim_const_map(params_dict,rX_vec,ri,T,mask_time,RF,tuned,map_seed,seeds,max_min,return_dynas=False,input_var=False):
    this_params_dict = params_dict.copy()
    this_params_dict['input_var'] = input_var
    this_params_dict['Sl'] = np.array([[params_dict['SlE'],params_dict['SlI']],[params_dict['SlE'],params_dict['SlI']]])
    this_params_dict['Sori'] = np.array([[params_dict['SoriE'],params_dict['SoriI']],[params_dict['SoriE'],params_dict['SoriI']]])
    this_params_dict['seed_con'] = int(map_seed)
    filtered_mydict_net = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
                                                    inspect.signature(network.network).parameters.values()]}
    net = network.network(**filtered_mydict_net)
    net.GI = this_params_dict['GI']
    if 'GE' in this_params_dict:
        net.GE = this_params_dict['GE']

    ori_idx=net.get_oriented_neurons(delta_ori=22.5)
    rf_idx=net.get_centered_neurons(stim_size=0.6)
    tune_idx=np.intersect1d(ori_idx,rf_idx)

    moments = {}
    preds_tuned = {}
    preds_untuned = {}
    rates = {}
    bals = {}
    optrs = {}
    if return_dynas:
        dynas = {}

    for rX_idx,this_rX in enumerate(rX_vec):
        print('Doing contrast '+str(rX_idx+1) +' of '+str(len(rX_vec)))

        this_params_dict['rX'] = this_rX

        rX_moments = np.zeros((len(seeds),10))
        rX_preds_tuned = np.zeros((len(seeds),6))
        rX_preds_untuned = np.zeros((len(seeds),6))
        rX_rates = np.zeros((len(seeds),2,6250))
        rX_bals = np.zeros((len(seeds),6250))
        rX_optrs = np.zeros((len(seeds),6250))
        if return_dynas:
            rX_dynas = np.zeros((len(seeds),2,6250,len(T)))

        for seed_idx,seed in enumerate(seeds):
            net.set_seed(int(seed))
            filtered_mydict_disorder = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
                                                inspect.signature(net.generate_disorder).parameters.values()]}
            net.generate_disorder(**filtered_mydict_disorder)

            ori_idx=net.get_oriented_neurons(delta_ori=22.5)
            rf_idx=net.get_centered_neurons(stim_size=0.6)
            tune_idx=np.intersect1d(ori_idx,rf_idx)

            if return_dynas:
                this_moments, _, _, this_rates, muEs, muIs, muLs, this_dynas = \
                    fun.get_moments_of_r_sim(net,ri,T,mask_time,this_params_dict['L'],RF,tuned,return_activity=True,
                        max_min=max_min,return_dynas=True)
            else:
                this_moments, _, _, this_rates, muEs, muIs, muLs = \
                    fun.get_moments_of_r_sim(net,ri,T,mask_time,this_params_dict['L'],RF,tuned,return_activity=True,
                        max_min=max_min,return_dynas=False)

            rX_moments[seed_idx,:] = this_moments
            rX_preds_tuned[seed_idx,:] = vu.get_predictions_from_moments_and_idxs(this_moments,vu.tuned_idxs)
            rX_preds_untuned[seed_idx,:] = vu.get_predictions_from_moments_and_idxs(this_moments,vu.untuned_idxs)
            rX_rates[seed_idx,:,:] = this_rates
            rX_bals[seed_idx,:] = np.abs(muEs+muIs)/muEs
            rX_optrs[seed_idx,:] = muLs/muEs
            if return_dynas:
                rX_dynas[seed_idx,:,:,:] = this_dynas
    
        mask = np.invert(np.any(rX_moments > 1e4,1))
        try:
            moments[rX_idx] = np.mean(rX_moments[mask,:],0)
            preds_tuned[rX_idx] = np.mean(rX_preds_tuned[mask,:],0)
            preds_untuned[rX_idx] = np.mean(rX_preds_untuned[mask,:],0)
            rates[rX_idx] = np.hstack([rX_rates[i,:,:] for i in np.arange(len(seeds))[mask]])
            bals[rX_idx] = np.concatenate([rX_bals[i,:] for i in np.arange(len(seeds))[mask]])
            optrs[rX_idx] = np.concatenate([rX_optrs[i,:] for i in np.arange(len(seeds))[mask]])
            if return_dynas:
                dynas[rX_idx] = np.hstack([rX_dynas[i,:,:,:] for i in np.arange(len(seeds))[mask]])
        except:
            moments[rX_idx] = 1e6*np.ones(10)
            preds_tuned[rX_idx] = 1e6*np.ones(6)
            preds_untuned[rX_idx] = 1e6*np.ones(6)
            rates[rX_idx] = 1e6*np.ones((2,6250))
            bals[rX_idx] = 1e6*np.ones(6250)
            optrs[rX_idx] = 1e6*np.ones(6250)
            if return_dynas:
                dynas[rX_idx] = 1e6*np.ones((2,6250,len(T)))
        moments[rX_idx][np.isnan(moments[rX_idx])==True] = 1e6
        preds_tuned[rX_idx][np.isnan(preds_tuned[rX_idx])==True] = 1e6
        preds_untuned[rX_idx][np.isnan(preds_untuned[rX_idx])==True] = 1e6
        rates[rX_idx][np.isnan(rates[rX_idx])==True] = 1e6
        bals[rX_idx][np.isnan(bals[rX_idx])==True] = 1e6
        optrs[rX_idx][np.isnan(optrs[rX_idx])==True] = 1e6
        if return_dynas:
            dynas[rX_idx][np.isnan(dynas[rX_idx])==True] = 1e6

    if return_dynas:
        return moments,preds_tuned,preds_untuned,rates,bals,optrs,dynas
    else:
        return moments,preds_tuned,preds_untuned,rates,bals,optrs


def sim_const_map_opto_switch(params_dict,rX_vec,ri,T,optoT,RF,tuned,map_seed,seeds,max_min,input_var=False):
    this_params_dict = params_dict.copy()
    this_params_dict['input_var'] = input_var
    this_params_dict['Sl'] = np.array([[params_dict['SlE'],params_dict['SlI']],[params_dict['SlE'],params_dict['SlI']]])
    this_params_dict['Sori'] = np.array([[params_dict['SoriE'],params_dict['SoriI']],[params_dict['SoriE'],params_dict['SoriI']]])
    this_params_dict['seed_con'] = int(map_seed)
    filtered_mydict_net = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
                                                    inspect.signature(network.network).parameters.values()]}
    net = network.network(**filtered_mydict_net)
    net.GI = this_params_dict['GI']
    if 'GE' in this_params_dict:
        net.GE = this_params_dict['GE']

    ori_idx=net.get_oriented_neurons(delta_ori=22.5)
    rf_idx=net.get_centered_neurons(stim_size=0.6)
    tune_idx=np.intersect1d(ori_idx,rf_idx)

    dynas = {}

    for rX_idx,this_rX in enumerate(rX_vec):
        print('Doing contrast '+str(rX_idx+1) +' of '+str(len(rX_vec)))

        this_params_dict['rX'] = this_rX

        rX_dynas = np.zeros((len(seeds),6250,len(T)))

        for seed_idx,seed in enumerate(seeds):
            net.set_seed(int(seed))
            filtered_mydict_disorder = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
                                                inspect.signature(net.generate_disorder).parameters.values()]}
            net.generate_disorder(**filtered_mydict_disorder)

            ori_idx=net.get_oriented_neurons(delta_ori=22.5)
            rf_idx=net.get_centered_neurons(stim_size=0.6)
            tune_idx=np.intersect1d(ori_idx,rf_idx)

            this_dynas = \
                fun.get_moments_of_r_sim_opto_switch(net,ri,T,optoT,this_params_dict['L'],RF,tuned,
                    max_min=max_min)

            rX_dynas[seed_idx,:,:] = this_dynas

        dynas[rX_idx] = np.hstack([rX_dynas[i,:,:] for i in np.arange(len(seeds))])

    return dynas


def sim_scan_params_avg_map(params_dict,rX_vec,param_scan_dict,ri,T,mask_time,RF,tuned,seeds,max_min):
    this_params_dict = params_dict.copy()

    params2vary = list(param_scan_dict.keys())
    nscanpars = len(params2vary)
    npsvs = [len(param_scan_dict[param]) for param in params2vary]
    npsv_iter = np.nditer(np.zeros(npsvs),flags=['multi_index'])

    varied_params = np.ndarray(npsvs,dtype=object)
    moments = np.zeros(npsvs+[len(rX_vec),10])
    preds_tuned = np.zeros(npsvs+[len(rX_vec),6])
    preds_untuned = np.zeros(npsvs+[len(rX_vec),6])
    bals = np.zeros(npsvs+[len(rX_vec),2])
    optrs = np.zeros(npsvs+[len(rX_vec),2])

    for _ in npsv_iter:
        npsv_idxs = npsv_iter.multi_index

        varied_params[npsv_idxs] = ''
        for idx,param in enumerate(params2vary):
            this_params_dict[param] = param_scan_dict[param][npsv_idxs[idx]]
            varied_params[npsv_idxs] += param + f'={param_scan_dict[param][npsv_idxs[idx]]:.6f},'
        varied_params[npsv_idxs] = varied_params[npsv_idxs][:-1]

        moments[npsv_idxs],preds_tuned[npsv_idxs],preds_untuned[npsv_idxs],bals[npsv_idxs],optrs[npsv_idxs] = \
            sim_avg_map(this_params_dict,rX_vec,ri,T,mask_time,RF,tuned,seeds,max_min)

    return varied_params,moments,preds_tuned,preds_untuned,bals,optrs


def sim_scan_params_const_map(params_dict,rX_vec,param_scan_dict,ri,T,mask_time,RF,tuned,map_seed,seeds,max_min,return_dynas=False):
    this_params_dict = params_dict.copy()

    params2vary = list(param_scan_dict.keys())
    nscanpars = len(params2vary)
    npsvs = [len(param_scan_dict[param]) for param in params2vary]
    npsv_iter = np.nditer(np.zeros(npsvs),flags=['multi_index'])

    varied_params = np.ndarray(npsvs,dtype=object)
    moments = {}
    preds_tuned = {}
    preds_untuned = {}
    rates = {}
    bals = {}
    optrs = {}
    if return_dynas:
        dynas = {}

    for _ in npsv_iter:
        npsv_idxs = npsv_iter.multi_index

        varied_params[npsv_idxs] = ''
        for idx,param in enumerate(params2vary):
            this_params_dict[param] = param_scan_dict[param][npsv_idxs[idx]]
            varied_params[npsv_idxs] += param + f'={param_scan_dict[param][npsv_idxs[idx]]:.6f},'
        varied_params[npsv_idxs] = varied_params[npsv_idxs][:-1]

        if return_dynas:
            moments[npsv_idxs],preds_tuned[npsv_idxs],preds_untuned[npsv_idxs],rates[npsv_idxs],bals[npsv_idxs],optrs[npsv_idxs],dynas[npsv_idxs] = \
                sim_const_map(this_params_dict,rX_vec,ri,T,mask_time,RF,tuned,map_seed,seeds,max_min,True)
        else:
            moments[npsv_idxs],preds_tuned[npsv_idxs],preds_untuned[npsv_idxs],rates[npsv_idxs],bals[npsv_idxs],optrs[npsv_idxs] = \
                sim_const_map(this_params_dict,rX_vec,ri,T,mask_time,RF,tuned,map_seed,seeds,max_min,False)

    if return_dynas:
        return varied_params,moments,preds_tuned,preds_untuned,rates,bals,optrs,dynas
    else:
        return varied_params,moments,preds_tuned,preds_untuned,rates,bals,optrs

def get_input_per_ori(net,this_RATES,nob=9):


    #######################################################
    # Excitatory or inhibitory matrix
    Is_excitatory=np.array([k in net.allE for k in range(net.N) ]*1)
    Is_inhibitory=np.array([k not in net.allE for k in range(net.N) ]*1)

    Is_excitatory_mat=np.matlib.repmat(Is_excitatory,len(Is_excitatory),1)
    Is_inhibitory_mat=np.matlib.repmat(Is_inhibitory,len(Is_inhibitory),1)


    #######################################################
    # Input Matrix
    Rates_mat_0=np.matlib.repmat(this_RATES[0,:],len(this_RATES[0,:]),1)
    Rates_mat_1=np.matlib.repmat(this_RATES[1,:],len(this_RATES[1,:]),1)

    Excitatory_Input_Mat_0=(net.M*Rates_mat_0*Is_excitatory_mat)
    Inhibitory_Input_Mat_0=(net.M*Rates_mat_0*Is_inhibitory_mat)

    Excitatory_Input_Mat_1=(net.M*Rates_mat_1*Is_excitatory_mat)
    Inhibitory_Input_Mat_1=(net.M*Rates_mat_1*Is_inhibitory_mat)

    Excitatory_Input_Mat_Diff=Excitatory_Input_Mat_1-Excitatory_Input_Mat_0
    Inhibitory_Input_Mat_Diff=Inhibitory_Input_Mat_1-Inhibitory_Input_Mat_0

    #######################################################
    # Difference in orientation matrix
    ZdMat = np.matlib.repmat(net.Z,net.Nloc*net.NT,1)
    ZDiffMat_nonnomr = np.abs(ZdMat-np.transpose(ZdMat))
    deltaZ = net.make_periodic(ZDiffMat_nonnomr,90)

    #######################################################

    ori_bounds=np.linspace(0,90,nob+1)

    Excitatory_input_per_ori=np.zeros((net.N,nob))
    Inhibitory_input_per_ori=np.zeros((net.N,nob))

    for k in range(nob):
        selected= (deltaZ<ori_bounds[k+1])*(ori_bounds[k]<= deltaZ)
        Excitatory_input_per_ori[:,k]=np.nansum(selected*Excitatory_Input_Mat_Diff,axis=1)
        Inhibitory_input_per_ori[:,k]=np.nansum(selected*Inhibitory_Input_Mat_Diff,axis=1)


    return Excitatory_input_per_ori, Inhibitory_input_per_ori

def get_all_input_per_ori(net,this_RATES,nob=18,signed=False):


    #######################################################
    # Excitatory or inhibitory matrix
    Is_excitatory=np.array([k in net.allE for k in range(net.N) ]*1)
    Is_inhibitory=np.array([k not in net.allE for k in range(net.N) ]*1)

    Is_excitatory_mat=np.matlib.repmat(Is_excitatory,len(Is_excitatory),1)
    Is_inhibitory_mat=np.matlib.repmat(Is_inhibitory,len(Is_inhibitory),1)


    #######################################################
    # Input Matrix
    Rates_mat_0=np.matlib.repmat(this_RATES[0,:],len(this_RATES[0,:]),1)
    Rates_mat_1=np.matlib.repmat(this_RATES[1,:],len(this_RATES[1,:]),1)

    Excitatory_Input_Mat_0=(net.M*Rates_mat_0*Is_excitatory_mat)
    Inhibitory_Input_Mat_0=(net.M*Rates_mat_0*Is_inhibitory_mat)

    Excitatory_Input_Mat_1=(net.M*Rates_mat_1*Is_excitatory_mat)
    Inhibitory_Input_Mat_1=(net.M*Rates_mat_1*Is_inhibitory_mat)

    Excitatory_Input_Mat_Diff=Excitatory_Input_Mat_1-Excitatory_Input_Mat_0
    Inhibitory_Input_Mat_Diff=Inhibitory_Input_Mat_1-Inhibitory_Input_Mat_0

    #######################################################
    # Difference in orientation matrix
    ZdMat = np.matlib.repmat(net.Z,net.Nloc*net.NT,1)
    ZDiffMat_nonnomr = ZdMat-np.transpose(ZdMat)
    if not signed:
        ZDiffMat_nonnomr = np.abs(ZDiffMat_nonnomr)
    deltaZ = net.make_periodic(ZDiffMat_nonnomr,90)

    #######################################################

    if signed:
        ori_bounds = np.linspace(-90,90,nob+1)[:-1]+90/nob
        ori_centers = 0.5*(ori_bounds+np.roll(ori_bounds,-1))
        ori_centers[-1] = 90
    else:
        ori_bounds = np.linspace(0,90,nob+1)
        ori_centers = 0.5*(ori_bounds[:-1]+ori_bounds[1:])

    E_base_input_per_ori=np.zeros((net.N,nob))
    I_base_input_per_ori=np.zeros((net.N,nob))

    E_opto_input_per_ori=np.zeros((net.N,nob))
    I_opto_input_per_ori=np.zeros((net.N,nob))

    E_diff_input_per_ori=np.zeros((net.N,nob))
    I_diff_input_per_ori=np.zeros((net.N,nob))

    for k in range(nob):
        if signed:
            selected= (deltaZ<np.roll(ori_bounds,-1)[k])*(ori_bounds[k]<= deltaZ)
        else:
            selected= (deltaZ<ori_bounds[k+1])*(ori_bounds[k]<= deltaZ)

        E_base_input_per_ori[:,k]=np.nansum(selected*Excitatory_Input_Mat_0,axis=1)
        I_base_input_per_ori[:,k]=np.nansum(selected*Inhibitory_Input_Mat_0,axis=1)

        E_opto_input_per_ori[:,k]=np.nansum(selected*Excitatory_Input_Mat_1,axis=1)
        I_opto_input_per_ori[:,k]=np.nansum(selected*Inhibitory_Input_Mat_1,axis=1)

        E_diff_input_per_ori[:,k]=np.nansum(selected*Excitatory_Input_Mat_Diff,axis=1)
        I_diff_input_per_ori[:,k]=np.nansum(selected*Inhibitory_Input_Mat_Diff,axis=1)

    return E_base_input_per_ori, I_base_input_per_ori, E_opto_input_per_ori, I_opto_input_per_ori,\
        E_diff_input_per_ori, I_diff_input_per_ori

def get_input_per_ori_dist_center(net,this_RATES,nob=18,diff_width=10):
    
    Excitatory_input_per_ori, Inhibitory_input_per_ori= get_get_input_per_ori(net,this_RATES,nob)

    Excitatory_input_per_ori_dist_center_mean=np.zeros((nob+1,nob))
    Inhibitory_input_per_ori_dist_center_mean=np.zeros((nob+1,nob))
    Excitatory_input_per_ori_dist_center_std=np.zeros((nob+1,nob))
    Inhibitory_input_per_ori_dist_center_std=np.zeros((nob+1,nob))


    for k in range(nob):
        for l in range(nob+1):
            this_neurons=net.get_neurons_at_given_ori_distance_to_grating(diff_width=diff_width, degs_away_from_center=ori_bounds[l], grating_orientation=None)
            Excitatory_input_per_ori_dist_center_mean[l,k]=np.mean(Excitatory_input_per_ori[:,k]*this_neurons)
            Inhibitory_input_per_ori_dist_center_mean[l,k]=np.mean(Inhibitory_input_per_ori[:,k]*this_neurons)
            Excitatory_input_per_ori_dist_center_std[l,k]=np.std(Excitatory_input_per_ori[:,k]*this_neurons)
            Inhibitory_input_per_ori_dist_center_std[l,k]=np.std(Inhibitory_input_per_ori[:,k]*this_neurons)

    return Excitatory_input_per_ori_dist_center_mean, Inhibitory_input_per_ori_dist_center_mean, Excitatory_input_per_ori_dist_center_std, Inhibitory_input_per_ori_dist_center_std




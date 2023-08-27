#!/usr/bin/python
import sys
import os
import argparse
import functions as fun
import pickle
import numpy as np
import gc
from scipy import  optimize


from scipy.interpolate import interp1d
from sklearn.metrics import r2_score

import pandas as pd
import network as network
import functions as fun
import functions as funsim
import time


def normalization_formula_single(c,opto,Rm,Ro,n,m,D,sig,S):
    out=Rm *(Ro +c**n +opto**m*D)/(sig+c**n+opto**m * S)
    return out

    
def normalization_formula(contrast_vec,opto,Rm,Ro,n,m,D,sig,S):
    out_baseline=normalization_formula_single(contrast_vec,0,Rm,Ro,n,m,D,sig,S)
    out_opto=normalization_formula_single(contrast_vec,opto,Rm,Ro,n,m,D,sig,S)
    return np.concatenate((out_baseline,out_opto)).flatten()

def normalization_formula_reduced(contrast_vec,Rm,Ro,n,D,sig,S):
    out_baseline=normalization_formula_single(contrast_vec,0,Rm,Ro,n,1,D,sig,S)
    out_opto=normalization_formula_single(contrast_vec,1,Rm,Ro,n,1,D,sig,S)
    return np.concatenate((out_baseline,out_opto)).flatten()


def normalization_formula_opt(params,contrast_vec,this_cell_curve,reduced=False):
    if reduced:
        [Rm,Ro,n,D,sig,S]=params
        out_baseline=normalization_formula_single(contrast_vec,0,Rm,Ro,n,1,D,sig,S)
        out_opto=normalization_formula_single(contrast_vec,1,Rm,Ro,n,1,D,sig,S)
    else:
        [opto,Rm,Ro,n,m,D,sig,S]=params
        out_baseline=normalization_formula_single(contrast_vec,0,Rm,Ro,n,m,D,sig,S)
        out_opto=normalization_formula_single(contrast_vec,opto,Rm,Ro,n,m,D,sig,S)
    return np.linalg.norm((np.concatenate((out_baseline,out_opto))-this_cell_curve))/np.linalg.norm(this_cell_curve)




def get_Nassi_guess(reduced):
    Ro_g=np.random.exponential(2)
    Rm_g=np.random.randn()*np.sqrt(5)+60
    sig_g=np.random.exponential(10)
    m_g=np.random.randn()*np.sqrt(0.5)+2
    n_g=np.random.randn()*np.sqrt(0.5)+2
    D_g=np.random.rand()*9.9+0.01
    S_g=np.random.rand()*9.9+0.01
    opto_g=np.random.rand()*5+0.01
    if reduced:
        return np.array([Rm_g,Ro_g,n_g,D_g,sig_g,S_g])
    else:
        return np.array([opto_g,Rm_g,Ro_g,n_g,m_g,D_g,sig_g,S_g])


def get_Agos_guess(this_cell_baseline):
    base_guess=np.array([ 0.7, 0.1, 1.2, 0.1,  1.54, 5.33, 0.1])
    [opto_g,Ro_g,n_g,m_g,D_g,sig_g,S_g]=base_guess*(2*np.random.rand(n_pars-1))
    Rm_g=this_cell_baseline*sig_g/Ro_g
    return np.array([opto_g,Rm_g,Ro_g,n_g,m_g,D_g,sig_g,S_g])


def get_normalization_curves(contrast,Rates_Matrix_flatten,ntrials_th=100,reduced=False):
    
    """
    Rates_Matrix_flatten : Matrix of n_cells x (contrast x 2 (laser points))
    """
    nc=len(contrast)
    n_cells_analyzed=Rates_Matrix_flatten.shape[0]
    #####################
    if reduced:
        n_pars=6
    else:
        n_pars=8
        
    all_pars=np.ones((n_cells_analyzed,n_pars))*np.nan
    all_fits=np.ones((n_cells_analyzed,2*nc))*np.nan
    R_sq=np.ones(n_cells_analyzed)*np.nan
    
    for this_cell in range(n_cells_analyzed):
        this_cell_curve=Rates_Matrix_flatten[this_cell,:]

        norm_vec=[]
        fitted_params_vec=[]

        for ntrials in range(ntrials_th):
            guess=get_Nassi_guess(reduced)
            result = optimize.minimize(normalization_formula_opt,guess,args=(contrast,this_cell_curve,reduced),method='Nelder-Mead')
            
            if result.success:
                norm=normalization_formula_opt(result.x,contrast,this_cell_curve,reduced)
                norm_vec.append(norm)
                fitted_params_vec.append(result.x)

        fitted_params=fitted_params_vec[np.argmin(norm_vec)]
        print('The relative error  for cell ' +str(this_cell) + ' is ' + str(norm_vec[np.argmin(norm_vec)]) )

        all_pars[this_cell,:]=fitted_params
        if reduced:
            all_fits[this_cell,:]=normalization_formula_reduced(contrast,*fitted_params)
        else:
            all_fits[this_cell,:]=normalization_formula(contrast,*fitted_params)
        res=all_fits[this_cell,:]-this_cell_curve
        sum_res=np.sum(res**2)
        ss_tot = np.sum((this_cell_curve-np.mean(this_cell_curve))**2)
        R_sq[this_cell]=1-sum_res/ss_tot
        
        #R_sq[this_cell]=r2_score(this_cell_curve, all_fits[this_cell,:])
        
    return all_pars,all_fits,R_sq




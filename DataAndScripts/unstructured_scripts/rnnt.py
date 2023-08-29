
import warnings
warnings.filterwarnings("ignore")
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.linalg import circulant
import scipy.integrate as integrate
from tqdm import tqdm
import scipy as sp
import os
import time
import random
import math
from scipy.optimize import fsolve
from copy import *


plt.rcParams.update({'font.size': 16})
from scipy import optimize
import sys


def rotate(a, n):
    l=list(a)
    return np.array(l[n:] + l[:n])

def isnan(vec):
    return np.mean(np.isnan(np.array(vec)))
    
    


###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

#  RNN Mean field


        


# This is to convert the gaussian dist of inputs to the firing rate distribution Eq 7/8 of roxin

def P_roxin_delta_sq(r,mu,delta_sq,xi):
    delta=np.sqrt(delta_sq)
    p_nu=(1/np.sqrt(2*np.pi)/xi/delta/r**(1-1/xi)*np.exp(-(r**(1/xi)-mu)**2/(2*delta**2)))*(r>0)
    p_nu[np.isnan(p_nu)]=0
    return p_nu+0.5*(1-erf(mu/np.sqrt(2)/delta))*(r==0);



def P_roxin_delta(r,mu,delta,xi):
    p_nu=(1/np.sqrt(2*np.pi)/xi/delta/r**(1-1/xi)*np.exp(-(r**(1/xi)-mu)**2/(2*delta**2)))*(r>0)
    p_nu[np.isnan(p_nu)]=0
    return p_nu+0.5*(1-erf(mu/np.sqrt(2)/delta))*(r==0);


###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
#  Roxin rate distribtion and kullback leibler divergences

def mean_roxin(mu,Delta,xi):
    return postitive_gaussian_moments(mu,Delta,xi)

def variance_roxin(mu,Delta,xi):
    second_moment= postitive_gaussian_moments(mu,Delta,2*xi)
    return second_moment-mean_roxin(mu,Delta,xi)**2

def mean_lognormal(mu,Delta):
    return np.exp(mu+Delta**2/2)
    
def variance_lognormal(mu,Delta):
    return (np.exp(Delta**2)-1)*np.exp(2*mu+Delta**2)









def f_prime_roxin(mu,Delta,xi):

    return xi*postitive_gaussian_moments(mu,Delta,(xi-1)/xi)


def postitive_gaussian_moments(mu,Delta,eta):
    term1=eta*mu*sp.special.gamma(eta/2)*sp.special.hyp1f1((1-eta)/2,3/2,-mu**2/2/Delta**2)
    term2=np.sqrt(2)*Delta*sp.special.gamma((1+eta)/2)*sp.special.hyp1f1(-eta/2,1/2,-mu**2/2/Delta**2)
    return 1/np.sqrt(np.pi)*2**((eta-3)/2)*Delta**(eta-1)*(term1+term2)




def log_likehood_roxin_fixed(parameters,input_dist,xi):
    # we write the -likelihood and minimize instead of maximizing
    mu = parameters[0]
    Delta = parameters[1]
    input_dist_positive=input_dist[input_dist>0]
    input_dist_zero=input_dist[input_dist==0]
    n_samples=len(input_dist)
    return (np.sum((input_dist_positive**(1/xi)-mu)**2/(2*Delta**2))+ np.sum(np.log(Delta)+np.log(xi)+1/2*np.log(2*np.pi)+(1-1/xi)*np.log(input_dist_positive))-sum(input_dist==0)*np.log(0.5*(1-erf(mu/np.sqrt(2*Delta**2)))))/n_samples

def maximum_log_likehood_roxin(input_dist):
    n_params=3
    success=False
    max_count=50
    c=0
    emp_mean=np.mean(input_dist)
    emp_std=np.std(input_dist)
    while not success and c<max_count :
        c+=1
        initial_guess = np.random.rand(n_params)
        result = sp.optimize.minimize(log_likehood_roxin, initial_guess,args=(input_dist))
        mean_diff = np.abs(mean_roxin(*result.x)-emp_mean)/emp_mean
        std_diff = np.abs(np.sqrt(variance_roxin(*result.x))-emp_std)/emp_std
        success=result.success
    if success:
        return result.x
    else :
        return result.x*np.nan

def log_likehood_roxin(parameters,input_dist):
    # we write the -likelihood and minimize instead of maximizing
    mu = parameters[0]
    Delta = parameters[1]
    xi = parameters[2]
    input_dist_positive=input_dist[input_dist>0]
    input_dist_zero=input_dist[input_dist==0]
    n_samples=len(input_dist)
    return (np.sum((input_dist_positive**(1/xi)-mu)**2/(2*Delta**2))+ np.sum(np.log(Delta)+np.log(xi)+1/2*np.log(2*np.pi)+(1-1/xi)*np.log(input_dist_positive)) - sum(input_dist==0)* np.log(0.5*(1-erf(mu/np.sqrt(2*Delta**2)))))/n_samples

def log_likehood_lognormal(parameters,input_dist):
    # we write the -likelihood and minimize instead of maximizing
    mu = parameters[0]
    Delta = parameters[1]
    input_dist_positive=input_dist[input_dist>0]
    input_dist_zero=input_dist[input_dist==0]
    n_samples=len(input_dist)
    return np.sum((np.log(input_dist_positive)-mu)**2/(2*Delta**2))+ np.sum(np.log(Delta)+1/2*np.log(2*np.pi)+np.log(input_dist_positive))


def maximum_log_likehood_roxin_fixed_xi(input_dist,xi=2):
    n_params=2
    success=False
    max_count=20
    c=0
    emp_mean=np.mean(input_dist)
    emp_std=np.std(input_dist)
    SC_guess=get_Roxin_params(np.mean(input_dist),np.std(input_dist)**2,xi)

    while not success and c< max_count:
        c+=1
        initial_guess = SC_guess+np.random.rand(n_params)
        result = sp.optimize.minimize(log_likehood_roxin_fixed, initial_guess,args=(input_dist,xi))
        mean_diff = np.abs(mean_roxin(*result.x,xi)-emp_mean)/emp_mean
        std_diff = np.abs(np.sqrt(variance_roxin(*result.x,xi))-emp_std)/emp_std
        success=result.success

    if success:
        return result.x
    else:
        return result.x*np.nan



def maximum_log_likehood_lognormal(input_dist):
    n_params=2
    success=False
    max_count=20
    c=0
    emp_mean=np.mean(input_dist)
    emp_std=np.std(input_dist)

    while not success and c< max_count:
        c+=1
        initial_guess = np.random.rand(n_params)*0.1
        result = sp.optimize.minimize(log_likehood_lognormal, initial_guess,args=(input_dist))
        mean_diff = np.abs(mean_lognormal(*result.x)-emp_mean)/emp_mean
        std_diff = np.abs(np.sqrt(variance_lognormal(*result.x))-emp_std)/emp_std
        success=result.success

    if success:
        return result.x
    else:
        return result.x*np.nan



def FindRoot_roxin(mudelta,mean_target,variance_target,xi):
    optoutput=np.zeros(2)
    optoutput[0]=mean_roxin(*mudelta,xi)-mean_target
    optoutput[1]=variance_roxin(*mudelta,xi)-variance_target
    return optoutput

    
def get_Roxin_params(mean_target,variance_target,xi=2):
    havesucc=False
    count=0
    while not havesucc and count<100:
        ic=[np.random.rand(2)*10]
        optoutput=optimize.root(FindRoot_roxin,ic, args=(mean_target,variance_target,xi),tol=1e-5);
        havesucc=optoutput.success
        count+=1
    return optoutput.x



def kullback_leibler_one_config(mu_d,delta_d,mu_mf,delta_mf): #Id the data and the model have same exponent, the KL doesnt depend on xi
    int_A=1/2*(1-erf(mu_d/np.sqrt(2)/delta_d))*np.log((1-erf(mu_d/np.sqrt(2)/delta_d))/(1-erf(mu_mf/np.sqrt(2)/delta_mf)))
    int_B=1/2*( erf(mu_d/np.sqrt(2)/delta_d)+1)*np.log(delta_mf/delta_d)
    int_C=1/4/delta_mf**2*((delta_d**2-delta_mf**2+(mu_d - mu_mf)**2)*(erf(mu_d/np.sqrt(2)/delta_d)+1) + np.sqrt(2/np.pi)/delta_d*np.exp(-mu_d**2/2/delta_d**2)*(delta_d**2*(mu_d-2*mu_mf)+delta_mf**2*mu_d))
    return int_A+int_B+int_C

def kullback_leibler_Roxins_forABC(mu_d,delta_d,mu_mf,var_mf, E_only_or_both= 'OnlyE'):
    nc=len(mu_d)
    KL=np.zeros(nc)
    for c in range(nc):
        if E_only_or_both=='OnlyE':
            KL[c]=kullback_leibler_one_config(mu_d[c],delta_d[c],mu_mf[c,0],np.sqrt(var_mf[c,0])) # we only have E data
        else:
            KL[c]=(kullback_leibler_one_config(mu_d[c],delta_d[c],mu_mf[c,0],np.sqrt(var_mf[c,0])) + kullback_leibler_one_config(mu_d[c],delta_d[c],mu_mf[c,1],np.sqrt(var_mf[c,1])))/2
            # we want E and I to be similary distributed so take both as part of the error.
        if np.isnan(KL[c]):
            break
    return np.linalg.norm(KL)




def IntegrandMean(ui,deltai,zi,rnpar):
    myintmean=rnnn.phi(ui+np.sqrt(deltai)*zi,rnpar)* np.exp(-zi**2/2)/np.sqrt(2*np.pi)
    return myintmean

def IntegrandSecondMoment(ui,deltai,zi,rnpar):
    myintvar=rnnn.phi(ui+np.sqrt(deltai)*zi,rnpar)**2 * np.exp(-zi**2/2)/np.sqrt(2*np.pi)
    return myintvar

def IntegralMean(u,delta,rnpar):
    return integrate.quad(lambda x: IntegrandMean(u,delta,x,rnpar), -np.inf,np.inf)[0]
#    return integrate.quad(lambda x: IntegrandMean(u,delta,x,rnpar), -100,100)[0]


def IntegralSecondMoment(u,delta,rnpar):
   return integrate.quad(lambda x: IntegrandSecondMoment(u,delta,x,rnpar),-np.inf,np.inf)[0]
#    return integrate.quad(lambda x: IntegrandSecondMoment(u,delta,x,rnpar),-100,100)[0]



def get_equivalence_mean_field(mult_factor_vec, W_mat, Var_mat,h_mean,h_var ,n):
    " to change the rates of each network by a factor given by mult_factor"
    
    
    W_factor=np.outer(mult_factor_vec**(1/n),1/mult_factor_vec)
    W_mat_new = W_mat*W_factor
    Var_mat_new=Var_mat*W_factor**2
    
    h_mean_new = h_mean*mult_factor_vec**(1/n)
    h_var_new = h_var*mult_factor_vec**(2/n)
    
    return W_mat_new, Var_mat_new,h_mean_new,h_var_new

    


def getNewSystem_Stronger_connectivity(Jmean,Jvar,fvec,h_mean_vanilla,h_var_vanilla,rnpar,Wdisttype,p,K, mult_factor):
    
    n=len(fvec)
    ndots=len(h_mean_vanilla)

    if Wdisttype=='sparse':
        this_ic_mf=np.random.rand(ndots,n**2)*0.2
        this_search_amplitud=10
    elif Wdisttype=='normal':
        this_ic_mf=np.random.rand(ndots,n**2)*0.2
        this_search_amplitud=10

    u_alpha, delta_sq_alpha=getMeanFieldContrastTuning_forABC(n,range(ndots),Jmean,Jvar,fvec,h_mean_vanilla, h_var_vanilla,\
                                                                   this_ic_mf,this_search_amplitud,rnpar,Wdisttype,p,K)

    
    Delta_h_mean=np.zeros_like(h_mean_vanilla)
    Delta_h_var=np.zeros_like(h_mean_vanilla)
    
    for c in range(ndots):
        Delta_h_mean[c,:],Delta_h_var[c,:]=FindEquivalentInputs_for_StrongerWmat(u_alpha[c,:],delta_sq_alpha[c,:],Jmean,Jvar,fvec,h_mean_vanilla, h_var_vanilla,\
                                                                                 rnpar,Wdisttype,p,K,mult_factor)
    
    final_Jmean=Jmean*mult_factor
    final_Jvar=Jvar*mult_factor**2
    final_h_mean=h_mean_vanilla+Delta_h_mean
    final_hvar=h_var_vanilla+Delta_h_var

    return final_Jmean, final_Jvar, final_h_mean, final_hvar
    





    
def getNewSystem_Stronger_connectivity(Jmean,Jvar,fvec,h_mean_vanilla,h_var_vanilla,rnpar,Wdisttype,p,K, mult_factor):
    
    n=len(fvec)
    ndots=len(h_mean_vanilla)

    if Wdisttype=='sparse':
        this_ic_mf=np.random.rand(ndots,n**2)*0.2
        this_search_amplitud=10
    elif Wdisttype=='normal':
        this_ic_mf=np.random.rand(ndots,n**2)*0.2
        this_search_amplitud=10

    u_alpha, delta_sq_alpha=getMeanFieldContrastTuning_forABC(n,range(ndots),Jmean,Jvar,fvec,h_mean_vanilla, h_var_vanilla,\
                                                                   this_ic_mf,this_search_amplitud,rnpar,Wdisttype,p,K)

    
    Delta_h_mean=np.zeros_like(h_mean_vanilla)
    Delta_h_var=np.zeros_like(h_mean_vanilla)
    
    for c in range(ndots):
        Delta_h_mean[c,:],Delta_h_var[c,:]=FindEquivalentInputs_for_StrongerWmat(u_alpha[c,:],delta_sq_alpha[c,:],Jmean,Jvar,fvec,h_mean_vanilla, h_var_vanilla,\
                                                                                 rnpar,Wdisttype,p,K,mult_factor)
    
    final_Jmean=Jmean*mult_factor
    final_Jvar=Jvar*mult_factor**2
    final_h_mean=h_mean_vanilla+Delta_h_mean
    final_hvar=h_var_vanilla+Delta_h_var

    return final_Jmean, final_Jvar, final_h_mean, final_hvar
    


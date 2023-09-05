#!/usr/bin/python
import sys

import argparse
#%load_ext autoreload
#%autoreload 2
#%matplotlib inline 
import matplotlib.pyplot as plt
import functions as fun

import numpy as np
import matplotlib as mpl
from scipy import stats


from scipy.interpolate import interp1d,interp2d
from scipy.optimize import least_squares

import pandas as pd


__author__ = 'sandro'
parser = argparse.ArgumentParser(description='This is a demo script by Sandro.')
parser.add_argument('-p','--params',help='Filename for results', required=True)
parser.add_argument('-r','--results',help='Filename for results', required=True)
parser.add_argument('-J', '--jobnumber', help='job number', required=False)
args = parser.parse_args()


print('Loading params file')
pdf = pd.read_csv('simulation_param.txt',delim_whitespace=True) #header=None
print(args)
jn = int(args.jobnumber)



sigma_t=pdf.sigma_t.iloc[jn]; 
tau_rp=pdf.tau_rp.iloc[jn]; 
g_E=pdf.g_E.iloc[jn]; 
g_I=pdf.g_I.iloc[jn]; 
beta=pdf.beta.iloc[jn]; 
K=int(pdf.K.iloc[jn]); 
sigma_Lambda_over_Lambda=pdf.sigma_Lambda_over_Lambda.iloc[jn]; 
J=pdf.J.iloc[jn]; 
r_X=pdf.r_X.iloc[jn];
ell=pdf.ell.iloc[jn];
Tmax_over_tau_E=int(pdf.Tmax_over_tau_E.iloc[jn]);

p=0.1; 
sigma_X_over_r_X=0.2; 
Lambda=1*10**-3; 

print('Parameters used')
print(sigma_t,tau_rp,g_E,g_I,beta,K,p,sigma_X_over_r_X,
      Lambda,sigma_Lambda_over_Lambda,J,r_X,ell,Tmax_over_tau_E)

r_X=[r_X]
L=[0,ell];
sigma_Lambda=sigma_Lambda_over_Lambda*Lambda
# In what follows, I compute W_{AB} starting from the parameters defined above
G_E,G_I=1.0,2.0 # Gain of Excitatory and inhibitory cells and I cells
w_EE=1;w_IE=w_EE/beta;
w_EI=g_E*w_EE;w_II=g_I*w_IE;
w_EX,w_IX=(G_I*fun.gamma*g_E-G_E)*w_EE,(G_I*fun.gamma*g_I-G_E)*w_IE; 
w_X=np.asarray([w_EX,w_IX]);
w=np.zeros((2,2));
w[0,:]=w_EE,-w_EI
w[1,:]=w_IE,-w_II

# I tabulate values of the single neuron f-I curve to spead up calculations below. 
mu_tab_max=10.0;
mu_tab=np.linspace(-mu_tab_max,mu_tab_max,200000)
mu_tab=np.concatenate(([-10000],mu_tab))
mu_tab=np.concatenate((mu_tab,[10000]))

phi_tab_E,phi_tab_I=mu_tab*0,mu_tab*0;
for idx in range(len(phi_tab_E)):
    phi_tab_E[idx]=fun.comp_phi_tab(mu_tab[idx],fun.tau_E,tau_rp,sigma_t)
    phi_tab_I[idx]=fun.comp_phi_tab(mu_tab[idx],fun.tau_I,tau_rp,sigma_t)

phi_int_E=interp1d(mu_tab, phi_tab_E, kind='linear')  
phi_int_I=interp1d(mu_tab, phi_tab_I, kind='linear')

# Generate quenched disorder
M,mu_X_over_r_X_tau,Lambda_i,N_E,N_I=fun.Generate_quenched_disorder(sigma_X_over_r_X,J,K,w,w_X,p,Lambda,sigma_Lambda)

print('Computing and saving network response')
T=np.arange(0,Tmax_over_tau_E*fun.tau_E,fun.tau_I/3);

mask_time=T>(10*fun.tau_E)
RATES=-1*np.ones((len(r_X),len(L),N_E+N_I))
DYNA=-1*np.ones((len(r_X),len(L),N_E+N_I,len(T)))
MUS=-1*np.ones((len(r_X),len(L),N_E+N_I))
Lambda_i_L=1./tau_rp*np.ones((len(r_X),len(L),N_E+N_I))
MFT_SOL_R=np.ones((len(r_X),len(L),10))
MFT_SOL_M=np.ones((len(r_X),len(L),10))
Phi=[phi_int_E,phi_int_I];
for idx_r_X in range(len(r_X)):
    for idx_L in range(len(L)):
        r_X_local,L_local=r_X[idx_r_X],L[idx_L];
        print((idx_r_X+1)/len(r_X),(idx_L+1)/len(L))
        DYNA[idx_r_X,idx_L,:,:], MUS[idx_r_X,idx_L,:],Lambda_i_L[idx_r_X,idx_L,:]=fun.High_dimensional_dynamics(T,L_local,r_X_local,M,mu_X_over_r_X_tau,Lambda_i,N_E,N_I,phi_int_E,phi_int_I);
        RATES[idx_r_X,idx_L,:]=np.mean(DYNA[idx_r_X,idx_L,:,mask_time],axis=0)
        print(np.mean(RATES[idx_r_X,idx_L,0:N_E]),
              np.mean(RATES[idx_r_X,idx_L,N_E::]),
              np.std(RATES[idx_r_X,idx_L,0:N_E]),
              np.std(RATES[idx_r_X,idx_L,N_E::]))


idx_r_X,idx_L=0,-1
Base_Sim=RATES[idx_r_X,0,:]
Delta_Sim=RATES[idx_r_X,idx_L,:]-RATES[idx_r_X,0,:]

moments_of_r_sim=np.zeros(5)
moments_of_r_sim[0]=np.mean(Base_Sim)
moments_of_r_sim[1]=np.mean(Delta_Sim)
moments_of_r_sim[2]=np.std(Base_Sim)
moments_of_r_sim[3]=np.std(Delta_Sim)
moments_of_r_sim[4]=np.cov(Base_Sim,Delta_Sim)[0,1]
print(moments_of_r_sim[:])


pippo_m=np.mean(DYNA[idx_r_X,idx_L,:,0:np.int32(len(T)/2)],axis=1)-np.mean(DYNA[idx_r_X,idx_L,:,np.int32(len(T)/2)::],axis=1)
pippo_p=np.mean(DYNA[idx_r_X,idx_L,:,0:np.int32(len(T)/2)],axis=1)+np.mean(DYNA[idx_r_X,idx_L,:,np.int32(len(T)/2)::],axis=1)

pippo_n=pippo_m/pippo_p

print('Saving results')
# simulations param+mean results+ meaurements of rate convergence

sim_param=np.asarray([sigma_t,tau_rp,g_E,g_I,beta,K,p,sigma_X_over_r_X,1,
      Lambda,sigma_Lambda_over_Lambda,J,r_X[idx_r_X],L[idx_L],Tmax_over_tau_E])
sim_results=moments_of_r_sim
sim_convergence=np.asarray([np.std(pippo_m),np.max(np.abs(pippo_m)),
                           np.std(pippo_n),np.max(np.abs(pippo_n))])
pippo_E=np.cov(Base_Sim[0:N_E],Delta_Sim[0:N_E])
pippo_I=np.cov(Base_Sim[N_E::],Delta_Sim[N_E::])

idx_r_X=0;idx_L=0;
idx_T_all=np.where((T>np.max(T)/2))[0]
auto=np.zeros(len(idx_T_all))
count=0;
for idx_T in idx_T_all:
    pippo=np.corrcoef(DYNA[idx_r_X,idx_L,:,idx_T],DYNA[idx_r_X,idx_L,:,idx_T_all[0]])
    auto[count]=pippo[0,1]
    count=count+1
max_decay=(np.max(T[idx_T_all])-np.min(T[idx_T_all]))
tau_decay=max_decay
idx_decay=np.argmin(np.abs(auto-0.5))
if np.min(auto)<0.95:
    tau_decay=T[0:len(auto)][idx_decay]
print(tau_decay,max_decay)    
    
    
additional_measurements=[pippo_E[0,1]/pippo_E[1,1],pippo_I[0,1]/pippo_I[1,1],tau_decay,max_decay]
results=np.zeros((1,len(sim_param)
                  +len(sim_results)
                 +len(sim_convergence)
                 +len(additional_measurements)))

results[0,0:len(sim_param)]=sim_param[:]
results[0,len(sim_param):(len(sim_param)+len(sim_results))]=sim_results
results[0,(len(sim_param)+len(sim_results)):(len(sim_param)+len(sim_results)+len(sim_convergence))]=sim_convergence
results[0,(len(sim_param)+len(sim_results)+len(sim_convergence))::]=additional_measurements

# Clean file to print results
f_handle = open('results/'+args.results,'w')
np.savetxt(f_handle,results,fmt='%.6f', delimiter='\t')
f_handle.close()

print('Done')










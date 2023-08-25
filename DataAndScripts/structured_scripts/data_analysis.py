
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


plt.rcParams.update({'font.size': 16})
from scipy import optimize
import sys

# mycmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
import rnnt
"#################################################################################################################"
"Data Class"
"#################################################################################################################"

N_stat=10**4;

class Data_MonkeyMouse(object):


    def __init__(self, animal,path2data,calc_delta_r=False):
    
        self.animal=animal
        self.path2data=path2data
        self.get_trial_data()
        self.get_distributions_raw(calc_delta_r=calc_delta_r)
        self.get_data()
        self.get_Roxin_mu_delta_sq_Vs_Con_Las('ML',xi=2)


    def get_trial_data_this_animal(self, this_animal):
    
        if this_animal=='monkey' or this_animal=='Monkey':
            data=np.loadtxt(self.path2data+'/Monkeys_with_trials.txt');

        elif this_animal=='mouse' or this_animal=='Mouse':
            data=np.loadtxt(self.path2data+'/Mice_with_trials.txt');
            
        elif this_animal=='both':
            data_m=np.loadtxt(self.path2data+'/Monkeys_with_trials.txt');
            data_M=np.loadtxt(self.path2data+'/Mice_with_trials.txt');
            data=[data_m,data_M]
            self.this_animals=['mouse','monkey']
        return data


    def get_trial_data(self):
    
        self.trial_data=self.get_trial_data_this_animal( self.animal)
    
    


    def get_data_per_animal(self,this_animal):
        
        contrast, laser, rates, rates_difference= self.get_distributions_raw_this_animal(this_animal)

        data_baseline_Vs_diffs_covariances=np.zeros_like(rates_difference[:,1:,0])
        data_baseline_Vs_Laser_covariances=np.zeros_like(rates_difference[:,1:,0])

        for this_c in range(len(contrast)):
            for this_l in range(1,len(laser)):
                this_rates=rates[this_c,0,:]
                this_rates_laser=rates[this_c,this_l,:]


                this_diffs=rates_difference[this_c,this_l,:]
                
                this_cov_diff=np.cov(this_rates[~np.isnan(this_rates)],this_diffs[~np.isnan(this_diffs)])
                data_baseline_Vs_diffs_covariances[this_c,this_l-1]=this_cov_diff[0,1]/this_cov_diff[0,0]
                
                this_cov_base_las=np.cov(this_rates[~np.isnan(this_rates)],this_rates_laser[~np.isnan(this_rates_laser)])
                data_baseline_Vs_Laser_covariances[this_c,this_l-1]=this_cov_base_las[0,1]/this_cov_base_las[0,0]
                    
        return data_baseline_Vs_diffs_covariances, data_baseline_Vs_diffs_covariances, rates.shape[-1]

    def get_data(self):

        if self.animal=='both':
            self.data_baseline_Vs_diffs_covariances=[]
            self.data_baseline_Vs_Laser_covariances=[]
            self.n_cells=[]

            for idx in range(len(self.this_animals)):
                data_baseline_Vs_diffs_covariances_aux,\
                data_baseline_Vs_Laser_covariances_aux, \
                n_cells_aux=self.get_data_per_animal(self.this_animals[idx])
                
                self.data_baseline_Vs_diffs_covariances.append(data_baseline_Vs_diffs_covariances_aux)
                self.data_baseline_Vs_Laser_covariances.append(data_baseline_Vs_Laser_covariances_aux)
                self.n_cells.append(n_cells_aux)
        else:
            self.data_baseline_Vs_diffs_covariances,\
            self.data_baseline_Vs_Laser_covariances, \
            self.n_cells=self.get_data_per_animal(self.animal)
            
            

    def get_bootstrap_moments(self,this_animal,calc_delta_r=False):
        
        data=self.get_trial_data_this_animal(this_animal)

            
        Con=np.unique(data[:,1])
        Las=np.unique(data[:,2])
        cells_id=np.unique(data[:,0]);
        Las=[Las[0],Las[-1]]

        Cell_Resp=np.zeros((len(cells_id),len(Con),len(Las),))
        for idx_cell in range(len(cells_id)):
            for idx_con in range(len(Con)):
                for idx_las in range(len(Las)):
                    mask=(data[:,0]==cells_id[idx_cell])&(data[:,2]==Las[idx_las])&(data[:,1]==Con[idx_con])
                    Trial_Resp=data[mask,3::]
                    Cell_Resp[idx_cell,idx_con,idx_las]=np.mean(Trial_Resp[np.isnan(Trial_Resp)==False])

        Bootstrap_idx=np.random.choice(np.arange(len(cells_id)),size=(len(cells_id),N_stat), replace=True)
        Bootstrap_Resp=np.zeros((N_stat,len(cells_id),len(Con),len(Las)))
        for idx_rep in range(N_stat):
            for idx_con in range(len(Con)):
                for idx_las in range(len(Las)):
                    Bootstrap_Resp[idx_rep,:,idx_con,idx_las]= Cell_Resp[[Bootstrap_idx[:,idx_rep]],idx_con,idx_las]

        if calc_delta_r:
            Moments=np.zeros((7,len(Con),2))
        else:
            Moments=np.zeros((6,len(Con),2))
        for idx_cases in range(len(Moments)):
            if (idx_cases==0)|(idx_cases==1):
                # mean  rates
                idx_las=idx_cases;
                Measurements=np.mean(Bootstrap_Resp[:,:,:,idx_las],axis=1)
            if (idx_cases==2)|(idx_cases==3):
                # std  rates
                idx_las=idx_cases-2;
                Measurements=np.std(Bootstrap_Resp[:,:,:,idx_las],axis=1)
            if (idx_cases==4):
                # std Delta rates
                Delta=Bootstrap_Resp[:,:,:,-1]-Bootstrap_Resp[:,:,:,0]
                Measurements=np.std(Delta,axis=1)
            if (idx_cases==5):
                # rho rates Delta rates
                Base=Bootstrap_Resp[:,:,:,0]
                Delta=Bootstrap_Resp[:,:,:,-1]-Bootstrap_Resp[:,:,:,0]
                Measurements=np.zeros((N_stat,len(Con)))
                for idx_rep in range(N_stat):
                    for idx_con in range(len(Con)):
                        pippo=np.cov(Base[idx_rep,:,idx_con],Delta[idx_rep,:,idx_con])
                        Measurements[idx_rep,idx_con]=pippo[0,1]/pippo[1,1]
            if (idx_cases==6):
                # Delta rates
                Delta=Bootstrap_Resp[:,:,:,-1]-Bootstrap_Resp[:,:,:,0]
                Measurements=np.mean(Delta,axis=1)
            
            for idx_con in range(len(Con)):
                Moments[idx_cases,idx_con,:]=np.mean(Measurements[:,idx_con]),np.std(Measurements[:,idx_con])

        return Moments
        
        


    def get_distributions_raw_this_animal(self,this_animal,calc_delta_r=False):

        data_monkeys=np.loadtxt(self.path2data+'/monkeys.txt');
        data_mouse=np.loadtxt(self.path2data+'/mice.txt');

        if this_animal=='monkey' or this_animal=='Monkey':
            data=data_monkeys

        elif this_animal=='mouse' or this_animal=='Mouse':
            data=data_mouse

        # Use only single units (MU are data[6,:]==0)
        data=data[data[:,6]==1,:]

        contrast=np.unique(data[:,1])
        laser=np.unique(data[:,2])
        N_cells=np.max(data[:,3])

        rates=np.zeros((len(contrast),len(laser),int(N_cells)))
        rates_diff=np.zeros((len(contrast),len(laser),int(N_cells)))


        for idx_C in range(len(contrast)):
            mask_no_laser=(data[:,1]==contrast[idx_C])&(data[:,2]==laser[0])
            Base=data[mask_no_laser,3]
            for idx_laser in range(len(laser)):
                mask_laser=(data[:,1]==contrast[idx_C])&(data[:,2]==laser[idx_laser])
                rates_aux=data[mask_laser,3]
                rates[idx_C,idx_laser,:]=np.concatenate((rates_aux,np.nan*np.zeros(int(N_cells-len(rates_aux)))))
                rates_diff_aux=rates_aux-Base
                rates_diff[idx_C,idx_laser,:]=np.concatenate((rates_diff_aux,np.nan*np.zeros(int(N_cells-len(rates_diff_aux)))))

        return contrast, laser, rates, rates_diff
        
        
    def get_distributions_raw(self,calc_delta_r=False):

        if self.animal =='both':
            contrast, nc, laser, rates, rates_diff, bootstrap_moments=[],[],[],[],[],[]
            for idx in range(len(self.this_animals)):
                contrast_aux, laser_aux, rates_aux, rates_diff_aux= self.get_distributions_raw_this_animal(self.this_animals[idx])
                bootstrap_moments_aux=self.get_bootstrap_moments(self.this_animals[idx],calc_delta_r)
                contrast.append(contrast_aux)
                nc.append(len(contrast_aux))
                laser.append(laser_aux)
                rates.append(rates_aux)
                rates_diff.append(rates_diff_aux)
                bootstrap_moments.append(bootstrap_moments_aux)
        else :
            contrast, laser, rates, rates_diff= self.get_distributions_raw_this_animal(self.animal)
            bootstrap_moments=self.get_bootstrap_moments(self.animal,calc_delta_r)
            nc=len(contrast)

        self.contrast=contrast
        self.nc=nc
        self.laser=laser
        self.rates=rates
        self.rates_difference=rates_diff
        self.bootstrap_moments=bootstrap_moments




    def get_Roxin_mu_delta_sq_singleC(self,This_dist,method,xi):
    
        This_dist=This_dist[~np.isnan(This_dist)]
        This_dist=This_dist[This_dist>0]

        if method=='SC':

            this_mu,this_delta=rnnt.get_Roxin_params(np.mean(This_dist),np.std(This_dist)**2,xi)
            this_xi=xi

        elif method=='ML':
            #This_dist[This_dist==0]=[]
            this_mu,this_delta=rnnt.maximum_log_likehood_roxin_fixed_xi(This_dist,xi)
            this_xi=xi
            if np.isnan(this_mu) or np.isnan(this_delta):
                this_mu,this_delta=rnnt.get_Roxin_params(np.mean(This_dist),np.std(This_dist)**2,xi)
        elif method=='MLXI':
            this_mu,this_delta,this_xi=rnnt.maximum_log_likehood_roxin(This_dist)

        return this_mu,this_delta,this_xi





    def get_Roxin_mu_delta_sq_Vs_Con_Las(self,method='SC',xi=2):


        if self.animal=='both':

            self.roxin_data_mu_matrix=[]
            self.roxin_data_delta_sq_matrix=[]
            self.roxin_data_xi_mat=[]
            

            for idx in range(len(self.this_animals)):
                this_rates=self.rates[idx]
                c_len=len(self.contrast[idx])
                l_len=len(self.laser[idx])


                roxin_data_mu_matrix=np.zeros((c_len,l_len))
                roxin_data_delta_sq_matrix=np.zeros((c_len,l_len))
                roxin_data_xi_mat=np.zeros((c_len,l_len))
        
                for c in range(c_len):
                    for l in range(l_len):
                    
                        This_dist=this_rates[c,l,:]
                        this_mu,this_delta,this_xi=self.get_Roxin_mu_delta_sq_singleC(This_dist,method,xi)

                        roxin_data_mu_matrix[c,l]=this_mu
                        roxin_data_delta_sq_matrix[c,l]=(this_delta)**2
                        roxin_data_xi_mat[c,l]=this_xi
                            
            self.roxin_data_mu_matrix.append(roxin_data_mu_matrix)
            self.roxin_data_delta_sq_matrix.append(roxin_data_delta_sq_matrix)
            self.roxin_data_xi_mat.append(roxin_data_xi_mat)
                            
                        
        else :
            this_rates=self.rates
            c_len=len(self.contrast)
            l_len=len(self.laser)
            roxin_data_mu_matrix=np.zeros((c_len,l_len))
            roxin_data_delta_sq_matrix=np.zeros((c_len,l_len))
            roxin_data_xi_mat=np.zeros((c_len,l_len))

            
            
            for c in range(c_len):
                for l in range(l_len):
                
                    This_dist=this_rates[c,l,:]
                    this_mu,this_delta,this_xi=self.get_Roxin_mu_delta_sq_singleC(This_dist,method,xi)

                    roxin_data_mu_matrix[c,l]=this_mu
                    roxin_data_delta_sq_matrix[c,l]=(this_delta)**2
                    roxin_data_xi_mat[c,l]=this_xi

                    
                    
            self.roxin_data_mu_matrix=roxin_data_mu_matrix
            self.roxin_data_delta_sq_matrix=roxin_data_delta_sq_matrix
            self.roxin_data_xi_mat=roxin_data_xi_mat
            


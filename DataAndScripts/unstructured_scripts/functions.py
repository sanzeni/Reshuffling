from numba import jit

import numpy as np
import scipy
from scipy.special import erf, erfi
from scipy.integrate import solve_ivp
from mpmath import fp
import network as network

import time

def get_moments_of_r_sim(net,ri,T,mask_time,L,return_activity=True,max_min=7.5, return_dynas=False):

    RATES=np.nan*np.ones((2,net.N))
    DYNAS=np.nan*np.ones((2,net.N,len(T)))
    MUS=np.nan*np.ones((2,net.N))
    MFT_SOL_R_A=np.ones((2,5))
    MFT_SOL_R_T=np.ones((2,5))
    MFT_SOL_M_A=np.ones((2,5))
    MFT_SOL_M_T=np.ones((2,5))

    # DYNAS[0,:,:],MUS[0,:]=\
        # high_dimensional_dynamics(net,ri,T,0.0);
    DYNAS[0,:,:],MUS[0,:],base_timeout=\
        high_dimensional_dynamics(ri,T,0.0,net.M,net.H,net.LAM,net.allE,net.allI,max_min);
    RATES[0,:]=np.mean(DYNAS[0,:,mask_time],axis=0)

    #print("Baseline Network with mean_E= {:.2f}, mean_I= {:.2f}, std_E= {:.2f}, std_I= {:.2f} "\
    #.format(np.mean(RATES[0,net.allE]),np.mean(RATES[0,net.allI]),np.std(RATES[0,net.allE]),np.std(RATES[0,net.allI])))

    
 
    # DYNAS[1,:,:],MUS[1,:]=\
        # high_dimensional_dynamics(net,ri,T,L);
    DYNAS[1,:,:],MUS[1,:],opto_timeout=\
        high_dimensional_dynamics(ri,T,L,net.M,net.H,net.LAM,net.allE,net.allI,max_min);
    RATES[1,:]=np.mean(DYNAS[1,:,mask_time],axis=0)


    #print("Optogenetically Stimulated Network with mean_E= {:.2f}, mean_I= {:.2f}, std_E= {:.2f}, std_I= {:.2f} "\
    #.format(np.mean(RATES[1,net.allE]),np.mean(RATES[1,net.allI]),np.std(RATES[1,net.allE]),np.std(RATES[1,net.allI])))
    
    
    
    mask_active=(RATES[0,:]>=0)|(RATES[1,:]>=0)
    Base_Sim=RATES[0,:][mask_active]
    Delta_Sim=(RATES[1,:]-RATES[0,:])[mask_active]

    moments_of_r_sim=np.zeros(5)

    if base_timeout:
        moments_of_r_sim[0]=1e6
        moments_of_r_sim[2]=1e6
    else:
        moments_of_r_sim[0]=np.mean(Base_Sim)
        moments_of_r_sim[2]=np.std(Base_Sim)

    if opto_timeout:
        moments_of_r_sim[1]=1e6
        moments_of_r_sim[3]=1e6
        moments_of_r_sim[4]=1e6
    else:
        moments_of_r_sim[1]=np.mean(Delta_Sim)
        moments_of_r_sim[3]=np.std(Delta_Sim)
        moments_of_r_sim[4]=np.cov(Base_Sim,Delta_Sim)[0,1]

    pippo_m=np.mean(DYNAS[1,:,0:np.int32(len(T)/2)],axis=1)-np.mean(DYNAS[1,:,np.int32(len(T)/2)::],axis=1)
    pippo_p=np.mean(DYNAS[1,:,0:np.int32(len(T)/2)],axis=1)+np.mean(DYNAS[1,:,np.int32(len(T)/2)::],axis=1)

    pippo_n=pippo_m/pippo_p

    pippo_E=np.cov(RATES[0,net.allE],RATES[1,net.allE]-RATES[0,net.allE])
    pippo_I=np.cov(RATES[0,net.allI],RATES[1,net.allI]-RATES[0,net.allI])

    idx_T_all=np.where((T>np.max(T)/2))[0]
    auto=np.zeros(len(idx_T_all))
    count=0;
    for idx_T in idx_T_all:
        pippo=np.corrcoef(DYNAS[0,:,idx_T],DYNAS[0,:,idx_T_all[0]])
        auto[count]=pippo[0,1]
        count=count+1
    max_decay=(np.max(T[idx_T_all])-np.min(T[idx_T_all]))
    tau_decay=max_decay
    idx_decay=np.argmin(np.abs(auto-0.5))
    if np.min(auto)<0.95:
        tau_decay=T[0:len(auto)][idx_decay]

    r_convergence = np.asarray([np.std(pippo_m),np.max(np.abs(pippo_m)),
                               np.std(pippo_n),np.max(np.abs(pippo_n))])
    r_pippo = np.asarray([pippo_E[0,1]/pippo_E[1,1],pippo_I[0,1]/pippo_I[1,1],tau_decay,max_decay])

    
    if return_activity or return_dynas:
        MUES = np.matmul(net.M[:,net.allE],DYNAS[0,net.allE,-1])+net.H
        MUIS = np.matmul(net.M[:,net.allI],DYNAS[0,net.allI,-1])
        MUES[net.allE],MUIS[net.allE] = ri.tau_E*MUES[net.allE],ri.tau_E*MUIS[net.allE]
        MUES[net.allI],MUIS[net.allI] = ri.tau_I*MUES[net.allI],ri.tau_I*MUIS[net.allI]
        MULS = net.LAM*L
        
    if return_dynas: #( this is the output format that is needed for the plots)
        return moments_of_r_sim, r_convergence, r_pippo, RATES, MUES, MUIS, MULS, DYNAS
    else:
        if return_activity:
            return moments_of_r_sim, r_convergence, r_pippo, RATES, MUES, MUIS, MULS
        else:
            return moments_of_r_sim, r_convergence, r_pippo

# def high_dimensional_dynamics(net,ri,T,L):
#     # This function compute the dynamics of the rate model
#     def system_RK45(t,R):
#         MU=np.matmul(net.M,R)+net.H
#         MU[net.allE]=ri.tau_E*MU[net.allE]
#         MU[net.allI]=ri.tau_I*MU[net.allI]
#         MU=MU+net.LAM*L
#         F=np.zeros(np.shape(MU))
#         F[net.allE] =(-R[net.allE]+ri.phi_int_E(MU[net.allE]))/ri.tau_E;
#         F[net.allI] =(-R[net.allI]+ri.phi_int_I(MU[net.allI]))/ri.tau_I;

#     RATES=np.zeros((net.N,len(T)));
#     sol=solve_ivp(system_RK45,[np.min(T),np.max(T)],RATES[:,0], method='RK45',t_eval=T)
#     RATES=sol.y;
    
#     MUS=np.matmul(net.M,RATES[:,-1])+net.H
#     MUS[net.allE]=ri.tau_E*MUS[net.allE]
#     MUS[net.allI]=ri.tau_I*MUS[net.allI]

#     return RATES,MUS

def high_dimensional_dynamics(ri,T,L,M,H,LAM,Einds,Iinds,max_min=7.5):
    F=np.zeros(np.shape(H))
    start = time.process_time()
    max_time = max_min*60
    timeout = False

    # This function computes the dynamics of the rate model
    def system_RK45(t,R):
        MU=np.matmul(M,R)+H
        MU[Einds]=ri.tau_E*MU[Einds]
        MU[Iinds]=ri.tau_I*MU[Iinds]
        MU=MU+LAM*L
        # F=np.zeros(np.shape(MU))
        F[Einds] =(-R[Einds]+ri.phi_int_E(MU[Einds]))/ri.tau_E;
        F[Iinds] =(-R[Iinds]+ri.phi_int_I(MU[Iinds]))/ri.tau_I;
        return F

    # This function determines if the system is stationary or not
    def stat_event(t,R):
        meanF = np.mean(np.abs(F)/np.maximum(R,1e-1)) - 5e-3
        if meanF < 0: meanF = 0
        return meanF
    stat_event.terminal = True

    # This function forces the integration to stop after 15 minutes
    def time_event(t,R):
        int_time = (start + max_time) - time.process_time()
        if int_time < 0: int_time = 0
        return int_time
    time_event.terminal = True

    RATES=np.zeros((len(H),len(T)));
    sol = solve_ivp(system_RK45,[np.min(T),np.max(T)],RATES[:,0], method='RK45', t_eval=T, events=[stat_event,time_event])
    if sol.t.size < len(T):
        print("      Integration stopped after " + str(np.around(T[sol.t.size-1],2)) + "s of simulation time")
        if time.process_time() - start > max_time:
            print("            Integration reached time limit")
            timeout = True
        RATES[:,0:sol.t.size] = sol.y
        RATES[:,sol.t.size:] = sol.y[:,-1:]
    else:
        RATES=sol.y;
    
    MU=np.matmul(M,RATES[:,-1])+H
    MU[Einds]=ri.tau_E*MU[Einds]
    MU[Iinds]=ri.tau_I*MU[Iinds]
    return RATES,MU,timeout


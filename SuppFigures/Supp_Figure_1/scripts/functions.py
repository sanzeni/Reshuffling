import numpy as np
import torch
from torchdiffeq import odeint, odeint_event
import scipy.integrate as integrate
import scipy
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.optimize import root
from scipy import stats
from scipy.optimize import curve_fit, minimize, least_squares
from scipy.special import erf
from scipy.optimize import fsolve

from scipy.integrate import quad, dblquad,tplquad
from scipy.stats import multivariate_normal
from scipy.integrate import solve_ivp


#tau_rp=2*10**(-3);
#tau_rp=10*10**(-3);

tau_E=0.02;tau_I=0.01;
theta=20*10**(-3);V_r=10*10**(-3);
gamma=1./4.;

tol=10**-3;  
N_stat=10**6;

def comp_phi_tab(mu,tau,tau_rp,sigma):
    nu_prova=np.linspace(0.,1.0/tau_rp,11)
    min_u=(V_r-mu)/sigma
    max_u=(theta-mu)/sigma
    r=np.zeros(np.size(mu));
    if np.size(mu)==1:
        if(max_u<10):            
            r=1.0/(tau_rp+tau*np.sqrt(np.pi)*integrale(min_u,max_u))
        if(max_u>=10):
            r=max_u/tau/np.sqrt(np.pi)*np.exp(-max_u**2)
    if np.size(mu)>1:
        for idx in range(len(mu)):
            if(max_u[idx]<10):            
                r[idx]=1.0/(tau_rp+tau*np.sqrt(np.pi)*integrale(min_u[idx],max_u[idx]))
            if(max_u[idx]>=10):
                r[idx]=max_u[idx]/tau/np.sqrt(np.pi)*np.exp(-max_u[idx]**2)
    return r

def integrale(minimo,massimo):
    if massimo<11:
        def f(x):
            param=-5.5
            if (x>=param):
                return np.exp(x**2)*(1+erf(x))
            if (x<param):
                return -1/np.sqrt(np.pi)/x*(1.0-1.0/2.0*pow(x,-2.0)+3.0/4.0*pow(x,-4.0))
        adelleh=integrate.quad(lambda u: f(u),minimo,massimo)
    if massimo>=11:
        adelleh=[1./massimo*np.exp(massimo**2)]
    return adelleh[0]



##################################################################
##### Compute network dynamics
##################################################################

def Generate_quenched_disorder(CV_K,J,K,w,w_X,p,Lambda,S_Lambda):
    # This function defines the quenced disorders in the problem: recurrent connectivity, feedforward inputs, opsin expression
    N=int(K/p);
    N_E,N_I=N,int(N*gamma);
    N_X=N_E+N_I

    # Define recurrent connectivity matrix
    #C=np.random.binomial(1,p,(N_E+N_I,N_E+N_I));
    C=np.zeros((N_E+N_I,N_E+N_I))
    
    K_E=K
    K_I=int(gamma*K)
    random_K=np.int32(np.random.normal(K_E, CV_K*K_E, N_E+N_I))
    random_K[random_K<1]=1
    for idx_post in range(N_E+N_I):
        possible_idx_pre=np.arange(0,N_E);
        mask=(possible_idx_pre!=idx_post);
        array_idx_pre=np.random.permutation(possible_idx_pre[mask])[:random_K[idx_post]];  #[0:K_E]#    
        C[idx_post,0:N_E][array_idx_pre]=1

    random_K=np.int32(np.random.normal(K_I, CV_K*K_I, N_E+N_I))
    random_K[random_K<0]=0
    for idx_post in range(N_E+N_I):
        possible_idx_pre=np.arange(0,N_I);
        mask=(possible_idx_pre!=idx_post);
        array_idx_pre=np.random.permutation(possible_idx_pre[mask])[:random_K[idx_post]];    #[0:K_I]#  
        C[idx_post,N_E::][array_idx_pre]=1


    M=np.zeros(np.shape(C))
    M[0:N_E,0:N_E],M[0:N_E,N_E::]=w[0,0]*C[0:N_E,0:N_E],w[0,1]*C[0:N_E,N_E::]
    M[N_E::,0:N_E],M[N_E::,N_E::]=w[1,0]*C[N_E::,0:N_E],w[1,1]*C[N_E::,N_E::]
    M=J*M

    # Define inputs normalized over r_X and tau_A
    mean_EX=K*J*w_X[0]
    mean_IX=K*J*w_X[1]
    
    s_X_over_r_X=0.2
    #var_EX=K**2*J**2*w_X[0]**2*(s_X_over_r_X**2/K+CV_K**2)#(mean_EX*CV_X)**2# K*J**2*w_X[0]**2*(s_X_over_r_X**2+(1-p))
    #var_IX=K**2*J**2*w_X[1]**2*(s_X_over_r_X**2/K+CV_K**2)#(mean_IX*CV_X)**2# K*J**2*w_X[1]**2*(s_X_over_r_X**2+(1-p))
    var_EX=K**2*J**2*w_X[0]**2*CV_K**2#(mean_EX*CV_X)**2# K*J**2*w_X[0]**2*(s_X_over_r_X**2+(1-p))
    var_IX=K**2*J**2*w_X[1]**2*CV_K**2#(mean_IX*CV_X)**2# K*J**2*w_X[1]**2*(s_X_over_r_X**2+(1-p))

    mu_X_over_r_X_tau=np.ones(N_X)
    mu_X_over_r_X_tau[0:N_E]=np.random.normal(mean_EX,np.sqrt(var_EX),(N_E))
    mu_X_over_r_X_tau[N_E::]=np.random.normal(mean_IX,np.sqrt(var_IX),(N_I))
    
    # Opsin expression: truncated Gaussian in a fraction frac of E cells, frac_effective represents the true fraction of cells with zero opsin expression
    
    sigma_Lambda_over_Lambda=S_Lambda/Lambda
    sigma_l=np.sqrt(np.log(1+sigma_Lambda_over_Lambda**2))
    mu_l=np.log(Lambda)-sigma_l**2/2

    Lambda_i=np.zeros(N_X)
    Lambda_i[0:N_E]=np.random.lognormal(mu_l, sigma_l, N_E)

    return M,mu_X_over_r_X_tau,Lambda_i,N_E,N_I;

def High_dimensional_dynamics(T,L,r_X,M,mu_X_over_r_X_tau,Lambda_i,N_E,N_I,phi_int_E,phi_int_I):
    # This function compute the dynamics of the rate model
    def system_RK45(t,R):
        MU=np.matmul(M,R)+mu_X_over_r_X_tau*r_X
        MU[0:N_E]=tau_E*MU[0:N_E]
        MU[N_E::]=tau_I*MU[N_E::]
        MU=MU+Lambda_i*L
        F=np.zeros(np.shape(MU))
        F[0:N_E] =(-R[0:N_E]+phi_int_E(MU[0:N_E]))/tau_E;
        F[N_E::] =(-R[N_E::]+phi_int_I(MU[N_E::]))/tau_I;
        return F

    RATES=np.zeros((N_E+N_I,len(T)));
    sol = solve_ivp(system_RK45,[np.min(T),np.max(T)],RATES[:,0], method='RK45',
            t_eval=T)
    RATES=sol.y;  
    
    MU=np.matmul(M,RATES[:,-1])+mu_X_over_r_X_tau*r_X
    MU[0:N_E]=tau_E*MU[0:N_E]
    MU[N_E::]=tau_I*MU[N_E::]
    return RATES,MU,Lambda_i*L;

def High_dimensional_dynamics(T,L,r_X,M,mu_X_over_r_X_tau,Lambda_i,N_E,N_I,phi_int_E,phi_int_I):
    # This function compute the dynamics of the rate model
    def system_RK45(t,R):
        MU=np.matmul(M,R)+mu_X_over_r_X_tau*r_X
        MU[0:N_E]=tau_E*MU[0:N_E]
        MU[N_E::]=tau_I*MU[N_E::]
        MU=MU+Lambda_i*L
        F=np.zeros(np.shape(MU))
        F[0:N_E] =(-R[0:N_E]+phi_int_E(MU[0:N_E]))/tau_E;
        F[N_E::] =(-R[N_E::]+phi_int_I(MU[N_E::]))/tau_I;
        return F

    RATES=np.zeros((N_E+N_I,len(T)));
    sol = solve_ivp(system_RK45,[np.min(T),np.max(T)],RATES[:,0], method='RK45',
            t_eval=T)
    RATES=sol.y;  
    
    MU=np.matmul(M,RATES[:,-1])+mu_X_over_r_X_tau*r_X
    MU[0:N_E]=tau_E*MU[0:N_E]
    MU[N_E::]=tau_I*MU[N_E::]
    return RATES,MU,Lambda_i*L;

def High_dimensional_dynamics_tensor(T,L,r_X,M,mu_X_over_r_X_tau,Lambda_i,E_cond,phi_int_E,phi_int_I):
    H = mu_X_over_r_X_tau*r_X
    LAS = Lambda_i*L

    # This function computes the dynamics of the rate model
    def system_RK45(t,R):
        MU=torch.matmul(M,R)#,out=MU)
        MU=torch.add(MU,H)#,out=MU)
        MU=torch.where(E_cond,tau_E*MU,tau_I*MU)#,out=MU)
        MU=torch.add(MU,LAS)#,out=MU)
        F=torch.where(E_cond,(-R+phi_int_E(MU))/tau_E,(-R+phi_int_E(MU))/tau_I)#,out=F)
        return F

    RATES=odeint(system_RK45,torch.zeros_like(mu_X_over_r_X_tau,dtype=torch.float32),T,method='rk4')
    
    MU=torch.matmul(M,RATES[-1,:])#,out=MU)
    MU=torch.add(MU,H)#,out=MU)
    MU=torch.where(E_cond,tau_E*MU,tau_I*MU)#,out=MU)
    return torch.transpose(RATES,0,1).cpu().numpy(),MU.cpu().numpy(),LAS.cpu().numpy()


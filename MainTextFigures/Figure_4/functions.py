import numpy as np
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
N_stat=10**4#10**6;

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

def Generate_quenched_disorder(s_X_over_r_X,J,K,w,w_X,p,Lambda,S_Lambda):
    # This function defines the quenced disorders in the problem: recurrent connectivity, feedforward inputs, opsin expression
    N=int(K/p);
    N_E,N_I=N,int(N*gamma);
    N_X=N_E+N_I

    # Define recurrent connectivity matrix
    C=np.random.binomial(1,p,(N_E+N_I,N_E+N_I));

    M=np.zeros(np.shape(C))
    M[0:N_E,0:N_E],M[0:N_E,N_E::]=w[0,0]*C[0:N_E,0:N_E],w[0,1]*C[0:N_E,N_E::]
    M[N_E::,0:N_E],M[N_E::,N_E::]=w[1,0]*C[N_E::,0:N_E],w[1,1]*C[N_E::,N_E::]
    M=J*M

    # Define inputs normalized over r_X and tau_A
    mean_EX=K*J*w_X[0]
    mean_IX=K*J*w_X[1]
    var_EX=K*J**2*w_X[0]**2*(s_X_over_r_X**2+(1-p))
    var_IX=K*J**2*w_X[1]**2*(s_X_over_r_X**2+(1-p))

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

###############################################################################################################################################################################################################################################################
# SOLVE MFT
###############################################################################################################################################################################################################################################################

def Statistics_of_phi_delta_phi(phi,moments_of_mu,opto):
    L,Lambda,S_Lambda=opto[:]
    mu,S_mu,delta_mu,S_delta_mu,Cov_mu_delta_mu=moments_of_mu[:]
    #Compute averages over inputs with delta

    Mean_Mat=[mu,delta_mu]
    Cov_Mat=np.zeros((2,2))
    Cov_Mat[0,0],Cov_Mat[1,1]=S_mu**2,S_delta_mu**2
    Cov_Mat[1,0],Cov_Mat[0,1]=Cov_mu_delta_mu,Cov_mu_delta_mu
    mu_i, delta_mu_i = np.random.multivariate_normal(Mean_Mat, Cov_Mat, N_stat).T
    
    lambda_i=np.zeros(np.shape(mu_i))
    if Lambda>0:
        sigma_Lambda_over_Lambda=S_Lambda/Lambda
        sigma_l=np.sqrt(np.log(1+sigma_Lambda_over_Lambda**2))
        mu_l=np.log(Lambda)-sigma_l**2/2
        lambda_i[:]=np.random.lognormal(mu_l, sigma_l, N_stat)
            
    mean_phi=np.mean(phi(mu_i))
    std_phi=np.std(phi(mu_i))
    mean_delta_phi=np.mean(phi(mu_i+delta_mu_i+lambda_i*L)-phi(mu_i))
    std_delta_phi=np.std(phi(mu_i+delta_mu_i+lambda_i*L)-phi(mu_i))
    Cov_phi_delta_phi=np.cov([phi(mu_i),(phi(mu_i+delta_mu_i+lambda_i*L)-phi(mu_i))])[0,1]

    return mean_phi,std_phi,mean_delta_phi,std_delta_phi,Cov_phi_delta_phi

def Solve_MF_equations(r_X,L,s_X_over_r_X,J,K,w,w_X,p,Lambda,S_Lambda,Phi):
    Key=np.asarray([K,gamma*K,K]);
    Tau=np.asarray([tau_E,tau_I]);
    s_X=s_X_over_r_X*r_X;

    W=np.zeros((2,3));W[:,0:2]=J*w[:,:];W[:,2]=J*w_X[:]

    def Predict_M_given_R(R):
        #R and M are ten dimensional vectors containing mean_F,std_F,mean_delta_F,std_delta_F,Cov_F_delta_F 
        # for E (first five components) and I (last five components) cells
        M=np.zeros(10);
        # mean M into E and I cells
        for idx_A in range(2):
            for idx_B in range(2):
                r=R[5*idx_B]
                Sr=R[5*idx_B+1]
                dr=R[5*idx_B+2]
                Sdr=R[5*idx_B+3]
                Crdr=R[5*idx_B+4]
                
                M[5*idx_A]=M[5*idx_A]+W[idx_A,idx_B]*Key[idx_B]*r;               
                M[5*idx_A+1]=M[5*idx_A+1]+W[idx_A,idx_B]**2*Key[idx_B]*((1-p)*r**2+Sr**2);
                M[5*idx_A+2]=M[5*idx_A+2]+W[idx_A,idx_B]*Key[idx_B]*dr;
                M[5*idx_A+3]=M[5*idx_A+3]+W[idx_A,idx_B]**2*Key[idx_B]*((1-p)*dr**2+Sdr**2);
                M[5*idx_A+4]=M[5*idx_A+4]+W[idx_A,idx_B]**2*Key[idx_B]*((1-p)*r*dr+Crdr);
                
            # Contribution from the X population only affect mooments without opto
            idx_B=2
            M[5*idx_A]=M[5*idx_A]+W[idx_A,idx_B]*Key[idx_B]*r_X;              
            M[5*idx_A+1]=M[5*idx_A+1]+W[idx_A,idx_B]**2*Key[idx_B]*((1-p)*r_X**2+s_X**2);
            
            M[5*idx_A+0]=Tau[idx_A]*M[5*idx_A+0]
            M[5*idx_A+1]=Tau[idx_A]*np.sqrt(M[5*idx_A+1])           
            M[5*idx_A+2]=Tau[idx_A]*M[5*idx_A+2]
            M[5*idx_A+3]=Tau[idx_A]*np.sqrt(M[5*idx_A+3])           
            M[5*idx_A+4]=Tau[idx_A]**2*M[5*idx_A+4]            
        return M

    def Predict_R_given_M(M):
        R=np.zeros(10);
        moments_of_mu=M[0:5]
        opto=[L,Lambda,S_Lambda]
        R[0:5]=Statistics_of_phi_delta_phi(Phi[0],moments_of_mu,opto)
        moments_of_mu=M[5::]
        opto=[0,0,0]
        R[5::]=Statistics_of_phi_delta_phi(Phi[1],moments_of_mu,opto)
        return R
    
    '''
    def system_RK45(t,M):
        F=np.zeros(np.shape(M))
        Predicted_R=Predict_R_given_M(M)
        Predicted_M=Predict_M_given_R(Predicted_R)
        
        F[0:5] =(-M[0:5]+Predicted_M[0:5])/tau_E;
        F[5::] =(-M[5::]+Predicted_M[5::])/tau_I;
        return F

    T=np.arange(0,20*tau_E,tau_I)
    sol = solve_ivp(system_RK45,[np.min(T),np.max(T)],np.zeros(10), method='RK45',
            t_eval=T)
    
    Predicted_M=sol.y[:,-1]
    Predicted_R=Predict_R_given_M(Predicted_M)
    Error_R=-Predicted_R+Predict_R_given_M(Predict_M_given_R(Predicted_R))
    return sol,Predicted_R,Predicted_M,Error_R
    '''
    def system_RK45(t,R):
        F=np.zeros(np.shape(R))
        Predicted_M=Predict_M_given_R(R)
        Predicted_R=Predict_R_given_M(Predicted_M)
        
        F[0:5] =(-R[0:5]+Predicted_R[0:5])/tau_E;
        F[5::] =(-R[5::]+Predicted_R[5::])/tau_I;
        return F

    T=np.arange(0,50*tau_E,tau_I)
    sol = solve_ivp(system_RK45,[np.min(T),np.max(T)],np.zeros(10), method='RK45',
            t_eval=T)
    
    Predicted_R=sol.y[:,-1]
    Predicted_M=Predict_M_given_R(Predicted_R)
    Error_R=-Predicted_R+Predict_R_given_M(Predicted_M)
    return sol,Predicted_R,Predicted_M,Error_R

def Distribution_of_phi_delta_phi(phi,moments_of_mu,opto):
    L,Lambda,S_Lambda=opto[:]
    mu,S_mu,delta_mu,S_delta_mu,Cov_mu_delta_mu=moments_of_mu[:]
    #Compute averages over inputs with delta

    Mean_Mat=[mu,delta_mu]
    Cov_Mat=np.zeros((2,2))
    Cov_Mat[0,0],Cov_Mat[1,1]=S_mu**2,S_delta_mu**2
    Cov_Mat[1,0],Cov_Mat[0,1]=Cov_mu_delta_mu,Cov_mu_delta_mu
    mu_i, delta_mu_i = np.random.multivariate_normal(Mean_Mat, Cov_Mat, N_stat).T
    
    lambda_i=np.zeros(np.shape(mu_i))
    if Lambda>0:
        sigma_Lambda_over_Lambda=S_Lambda/Lambda
        sigma_l=np.sqrt(np.log(1+sigma_Lambda_over_Lambda**2))
        mu_l=np.log(Lambda)-sigma_l**2/2
        lambda_i[:]=np.random.lognormal(mu_l, sigma_l, N_stat)
            
    mean_phi=np.mean(phi(mu_i))
    std_phi=np.std(phi(mu_i))
    mean_delta_phi=np.mean(phi(mu_i+delta_mu_i+lambda_i*L)-phi(mu_i))
    std_delta_phi=np.std(phi(mu_i+delta_mu_i+lambda_i*L)-phi(mu_i))
    Cov_phi_delta_phi=np.cov([phi(mu_i),(phi(mu_i+delta_mu_i+lambda_i*L)-phi(mu_i))])[0,1]

    return phi(mu_i),phi(mu_i+delta_mu_i+lambda_i*L)

'''
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
'''


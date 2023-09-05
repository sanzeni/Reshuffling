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
##### Distribution of opsin expression
##################################################################

def Gauss(x,mean,std):
    return 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*std**2))

def P_opsin_pstv(x,mean,std,frac):
    pippo=np.zeros(np.size(x))
    if np.size(x)>1:
        pippo=frac*(Gauss(x,mean,std))
    if np.size(x)==1:
        pippo=frac*(Gauss(x,mean,std))
    return pippo
def P_opsin_zero(x,mean,std,frac):
    pippo=np.zeros(np.size(x))
    if np.size(x)>1:
        pippo=1-frac/2+frac/2*(scipy.special.erf((0.-mean)/std/np.sqrt(2)))
    if np.size(x)==1:
        pippo=1-frac/2+frac/2*(scipy.special.erf((0.-mean)/std/np.sqrt(2)))
    return pippo
def P_opsin(x,mean,std,frac):
    pippo=np.zeros(np.size(x))
    if np.size(x)>1:
        pippo[x>0]=P_opsin_pstv(x[x>0],mean,std,frac)
        pippo[x==0]=P_opsin_zero(x[x==0],mean,std,frac)
    if np.size(x)==1:
        if x>0:
            pippo=P_opsin_pstv(x,mean,std,frac)
        if x==0:
            pippo=P_opsin_zero(x,mean,std,frac)
    return pippo

##################################################################
##### Compute averages over inputs
##################################################################

def mean_and_std_of_F(F,mu,S_mu,L,Lambda,S_Lambda,frac):
    lambda_i=np.random.normal(Lambda,S_Lambda, N_stat)
    lambda_i[int(frac*N_stat)::]=0;
    lambda_i[lambda_i<0]=0;
    mu_i=np.random.normal(mu,S_mu, N_stat)
    
    mean=np.mean(F(mu_i+lambda_i*L))
    std=np.std(F(mu_i+lambda_i*L))

    return mean,std


##################################################################
##### Solve implicit equation for momemens of rates
##################################################################
def solution_MFT(r_X,L,s_X_over_r_X,J,K,w,w_X,p,Lambda,S_Lambda,frac,Phi):
    Frac=[frac,0];
    Key=np.asarray([K,gamma*K,K])
    Tau=np.asarray([tau_E,tau_I])
    s_X=s_X_over_r_X*r_X

    W=np.zeros((2,3))
    W[:,0:2]=J*w[:,:]
    W[:,2]=J*w_X[:]
    
    def M_of_X(X):
        # R= six rows vector corresponding to r_E,r_I,r_X,\sigma_{r_E},\sigma_{r_I},\sigma_{r_X}
        # M= four rows vector corresponding to \mu_E,\mu_I,\sigma_{\mu_E},\sigma_{\mu_I}
        R=np.zeros(6);R[0:2]=X[0:2];R[2]=r_X;R[3:5]=X[2:4];R[5]=s_X;
        M=np.zeros(4);
        for idx_A in range(2):
            for idx_B in range(3):
                M[idx_A]=M[idx_A]+W[idx_A,idx_B]*Key[idx_B]*R[idx_B];               
                M[idx_A+2]=M[idx_A+2]+W[idx_A,idx_B]**2*Key[idx_B]*((1-p)*R[idx_B]**2+R[idx_B+3]**2);
            M[idx_A]=Tau[idx_A]*M[idx_A]
            M[idx_A+2]=Tau[idx_A]*np.sqrt(M[idx_A+2])
        return M
    def implicit(X):
        # R= six rows vector corresponding to r_E,r_I,r_X,\sigma_{r_E},\sigma_{r_I},\sigma_{r_X}
        # M= four rows vector corresponding to \mu_E,\mu_I,\sigma_{\mu_E},\sigma_{\mu_I}
        M=M_of_X(X);        
        errors=np.zeros(np.size(X));
        for idx_A in range(2):
            F,mu,S_mu=Phi[idx_A],M[idx_A],M[idx_A+2]
            errors[idx_A],errors[idx_A+2]=mean_and_std_of_F(F,mu,S_mu,L,Lambda,S_Lambda,Frac[idx_A])
            errors[idx_A],errors[idx_A+2]=errors[idx_A]-X[idx_A],errors[idx_A+2]-X[idx_A+2]
        #print(errors)
        return errors
    none,none,r_O0,r_E0,r_I0=    model_rates(np.asarray([r_X]),np.asarray([L]),J,K,w,w_X,Lambda,frac,Phi[0],Phi[1]);
    #R_0=[(frac)*r_O0+(1-frac)*r_E0,r_I0,0.5*((frac)*r_O0+(1-frac)*r_E0),0.5*r_I0];
    R_0=[(frac)*r_O0+(1-frac)*r_E0,r_I0,1.0*((frac)*r_O0+(1-frac)*r_E0),1.*r_I0];
    print(R_0)
    sol=fsolve(implicit,R_0,epsfcn=10**-5,xtol=10**-8,full_output=1)
    #sol=fsolve(implicit,R_0,epsfcn=10**-5,xtol=10**-6,full_output=1)
    #print(sol)
    sol = sol[0]
    
    sol[2::]=np.abs(sol[2::])
    return sol,M_of_X(sol),np.sqrt(np.sum(implicit(sol)**2))



##################################################################
##### Compute solutions without disorder
##################################################################
def mu(nu_O,nu_E,nu_I,W_AE,W_AI,frac,I_AX):
    return W_AE*frac*nu_O+W_AE*(1-frac)*nu_E+W_AI*nu_I+I_AX

def implicit(nu_O,nu_E,nu_I,L,Ws,Is,Lambda,frac,phi_int_E,phi_int_I):
    # No quenched disorder
    W_EE,W_EI,W_IE,W_II=Ws[0:4]
    I_EX,I_IX=Is[0:2]
    mu_O=mu(nu_O,nu_E,nu_I,W_EE,W_EI,frac,I_EX)+Lambda*L
    mu_E=mu(nu_O,nu_E,nu_I,W_EE,W_EI,frac,I_EX)
    mu_I=mu(nu_O,nu_E,nu_I,W_IE,W_II,frac,I_IX)
    ops=-nu_O+phi_int_E(mu_O) 
    ecc=-nu_E+phi_int_E(mu_E) 
    inh=-nu_I+phi_int_I(mu_I)
    return (ops,ecc,inh)
            
def model_rates(r_X,L,J,K,w,w_X,Lambda,frac,phi_int_E,phi_int_I):
    # This function computes the response of Excitatory and Inhibitory cells in the absence of quenched disorder
    nu_prova=np.linspace(0.,phi_int_E(1),11)
    W_EE,W_EI,W_IE,W_II=tau_E*J*K*w[0,0],tau_E*J*K*gamma*w[0,1],tau_I*J*K*w[1,0],tau_I*J*K*gamma*w[1,1]
    I_EX,I_IX=w_X[0]*tau_E*J*K*r_X,w_X[1]*tau_I*J*K*r_X
    local_Ws=[W_EE,W_EI,W_IE,W_II]
    DATA=np.zeros((1,5))
    for idx_X in range(np.size(r_X)):
        local_Is=[I_EX[idx_X],I_IX[idx_X]]
        def Fun(x):
            err=implicit(x[0],x[1],x[2],L[idx_X],local_Ws,local_Is,Lambda,frac,phi_int_E,phi_int_I)
            return [err[0],err[1],err[2]]
        data=2*1./0.002*np.ones((1,5));
        data[0,0]=r_X[idx_X];
        data[0,1]=L[idx_X];
    
        count=0.0;
        for i in range(len(nu_prova)):
            for ii in range(len(nu_prova)):
                for iii in range(len(nu_prova)):
                    pippo=0
                    try:
                        sol = scipy.optimize.root(Fun, [nu_prova[i],nu_prova[ii],nu_prova[iii]], method='hybr')
                        pippo=1
                        solution=np.asarray([r_X[idx_X],L[idx_X],sol.x[0],sol.x[1],sol.x[2]])
                    except:
                        pass
                    if(pippo>0):
                        if (sol.success==True)&(sol.x[0]>=0)&(sol.x[1]>=0)&(sol.x[2]>=0)&((np.sum(sol.fun**2))<tol):
                            if count>0:
                                d=100
                                for ind in range(np.shape(data)[0]-1):
                                    d=min(np.sqrt(np.sum((data[ind+1]-solution)**2)),d)
                                if d>5.0:  
                                    data=np.vstack((data, solution))
                                    count=count+1
                            if count==0:
                                data=np.vstack((data, solution))
                                count=count+1

        if np.shape(data)[0]==1:
            DATA=np.vstack((DATA, data))
        if(np.shape(data)[0]>1):
            data= np.delete(data, (0), axis=0)
            DATA=np.vstack((DATA, data))
    DATA= np.delete(DATA, (0), axis=0)
    return DATA[:,0],DATA[:,1],DATA[:,2],DATA[:,3],DATA[:,4]



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

##################################################################
##### Compute averages over inputs with delta
##################################################################

def Statistics_of_F_delta_F(F,mu,S_mu,delta_mu,S_delta_mu,Cov_mu_delta_mu,L,Lambda,S_Lambda,frac):
    # F is a function of mu^i+\Delta mu^i+Lambda^i*L
    # I compute the average separating the component with Lambda^i=0
    
    lambda_i=np.random.normal(Lambda,S_Lambda, N_stat)
    lambda_i[int(frac*N_stat)::]=0;
    lambda_i[lambda_i<0]=0;
    mu_i=np.random.normal(mu,S_mu, N_stat)
    
    Mean_Mat=[mu,delta_mu]
    Cov_Mat=np.zeros((2,2))
    Cov_Mat[0,0],Cov_Mat[1,1]=S_mu**2,S_delta_mu**2
    Cov_Mat[1,0],Cov_Mat[0,1]=Cov_mu_delta_mu,Cov_mu_delta_mu
    mu_i, delta_mu_i = np.random.multivariate_normal(Mean_Mat, Cov_Mat, N_stat).T
    
    mean_F=np.mean(F(mu_i))
    std_F=np.std(F(mu_i))
    mean_delta_F=np.mean(F(mu_i+delta_mu_i+lambda_i*L)-F(mu_i))
    std_delta_F=np.std(F(mu_i+delta_mu_i+lambda_i*L)-F(mu_i))
    Cov_F_delta_F=np.cov([F(mu_i),(F(mu_i+delta_mu_i+lambda_i*L)-F(mu_i))])[0,1]

    return mean_F,std_F,mean_delta_F,std_delta_F,Cov_F_delta_F


##################################################################
##### Solve implicit equation for momemens of rates  with delta
##################################################################
def Cov_solution_MFT_with_delta(r_X,L,s_X_over_r_X,J,K,w,w_X,p,Lambda,S_Lambda,frac,Phi,Aux,R_0):
    # Aux is list which contained the auxiliary variable needed to compute the covariance.
    r_E,r_I,sigma_r_E,sigma_r_I=Aux[0:4]
    rates=[r_E,r_I]
    mu_E,mu_I,sigma_mu_E,sigma_mu_I=Aux[4:8]
    p=Aux[8]

    Frac=[frac,0];Key=np.asarray([K,gamma*K,K]);
    Tau=np.asarray([tau_E,tau_I]);s_X=s_X_over_r_X*r_X;

    W=np.zeros((2,3));W[:,0:2]=J*w[:,:];W[:,2]=J*w_X[:]
    MU=[mu_E,mu_I];
    S_MU=[sigma_mu_E,sigma_mu_I]
    
    def M_of_X(X):
        #X= six rows vector corresponding to 
        #\delta  r_E,\delta r_I, \sigma^2_{\delta  r_E},\sigma^2_{\delta r_I}, Cov r_E \delta r_E, Cov r_I \delta r_I  
        #M= six rows vector corresponding to 
        #\delta \mu_E,\delta \mu_I, sigma^2_{\delta \mu_E},\sigma^2_{\delta \mu_I}, Cov \mu_E \delta \mu_E, Cov\mu_I \delta \mu_I,
        M=np.zeros(6);
        for idx_A in range(2):
            for idx_B in range(2): # No contribution from the X populations
                M[idx_A]=M[idx_A]+W[idx_A,idx_B]*Key[idx_B]*X[idx_B];
                M[idx_A+2]=M[idx_A+2]+W[idx_A,idx_B]**2*Key[idx_B]*((1-p)*X[idx_B]**2+X[idx_B+2]);
                #M[idx_A+4]=M[idx_A+4]+W[idx_A,idx_B]**2*Key[idx_B]*X[idx_B+4];
                M[idx_A+4]=M[idx_A+4]+W[idx_A,idx_B]**2*Key[idx_B]*(X[idx_B+4]+(1-p)*rates[idx_B]*X[idx_B])
            M[idx_A]=Tau[idx_A]*M[idx_A]
            M[idx_A+2]=Tau[idx_A]**2*M[idx_A+2]
            M[idx_A+4]=Tau[idx_A]**2*M[idx_A+4]
        return M

    def Predict_Rates(M):
        #M= six rows vector corresponding to 
        #\delta \mu_E,\delta \mu_I, sigma^2_{\delta \mu_E},\sigma^2_{\delta \mu_I}, Cov \mu_E \delta \mu_E, Cov\mu_I \delta \mu_I,
        R=np.zeros(6);
        for idx_A in range(2):
            F,mu,S_mu,delta_mu=Phi[idx_A],MU[idx_A],S_MU[idx_A],M[idx_A]
            S2_delta_mu,Cov_mu_delta_mu=M[idx_A+2],M[idx_A+4]
            pippo=Statistics_of_F_delta_F(F,mu,S_mu,delta_mu,np.sqrt(S2_delta_mu),Cov_mu_delta_mu,
                                          L,Lambda,S_Lambda,Frac[idx_A])
            R[idx_A],R[idx_A+2],R[idx_A+4]=pippo[2],pippo[3]**2,pippo[4]
            #print(pippo)#mean_F,std_F,mean_delta_F,std_delta_F,Cov_F_delta_F
        return R
    
    def implicit(X):
        #X= six rows vector corresponding to 
        #\delta  r_E,\delta r_I, \sigma^2_{\delta  r_E},\sigma^2_{\delta r_I}, Cov r_E \delta r_E, Cov r_I \delta r_I
        X[2],X[3]=np.abs(X[2]),np.abs(X[3])
        M=M_of_X(X);  
        R=Predict_Rates(M);
        errors=np.zeros(np.size(X));
        for idx_A in range(2):
            errors[idx_A]=R[idx_A]-X[idx_A]
            errors[idx_A+2]=np.sqrt(R[idx_A+2])-np.sqrt(X[idx_A+2])
            errors[idx_A+4]=np.sign(R[idx_A+4])*np.sqrt(np.abs(R[idx_A+4]))-np.sign(X[idx_A+4])*np.sqrt(np.abs(X[idx_A+4])) 
        #print(errors)
        return errors

    sol=fsolve(implicit,R_0,epsfcn=10**-5,xtol=10**-6,full_output=1)
    #sol=fsolve(implicit,R_0,epsfcn=10**-5,xtol=10**-4,full_output=1)
    #print(sol)
    sol = sol[0]
    return sol,M_of_X(sol),np.sqrt(np.sum(implicit(sol)**2))
    #'''
    


def solution_MFT_with_delta(r_X,L,s_X_over_r_X,J,K,w,w_X,p,Lambda,S_Lambda,frac,Phi):
   
    # 1) Compute r and sigma_r without laser
    pippo=solution_MFT(r_X,L,s_X_over_r_X,J,K,w,w_X,p,Lambda,S_Lambda,0,Phi)
    r_E,r_I,sigma_r_E,sigma_r_I=pippo[0][:]
    mu_E,mu_I,sigma_mu_E,sigma_mu_I=pippo[1][:]    
    print(r_E,r_I,sigma_r_E,sigma_r_I)

    # 2) Solve the implicit conditions with L to compute initial conditions in the next step
    pippo=solution_MFT(r_X,L,s_X_over_r_X,J,K,w,w_X,p,Lambda,S_Lambda,frac,Phi)
    r_E_L,r_I_L,sigma_r_E_L,sigma_r_I_L=pippo[0][:]

    R_0=[r_E_L-r_E,r_I_L-r_I,
         np.max([sigma_r_E_L**2-sigma_r_E**2,0]),np.max([sigma_r_I_L**2-sigma_r_I**2,0]),
         np.min([sigma_r_E_L**2-sigma_r_E**2,0]),np.min([sigma_r_I_L**2-sigma_r_I**2,0]),];
    print(R_0)
    # 2) Solve the implicit conditions for variance of Delta_r and covariance r Delta_r
    Aux=[r_E,r_I,sigma_r_E,sigma_r_I,mu_E,mu_I,sigma_mu_E,sigma_mu_I,p]

    
    sol_R,sol_M,err=Cov_solution_MFT_with_delta(
        r_X,L,s_X_over_r_X,J,K,w,w_X,p,Lambda,S_Lambda,frac,Phi,Aux,R_0)
    
    delta_r_E,delta_r_I=sol_R[0:2]
    S2_delta_r_E,S2_delta_r_I,cov_r_E,cov_r_I=sol_R[2:6]
    sol_R=[r_E,r_I,sigma_r_E**2,sigma_r_I**2,delta_r_E,delta_r_I,
           S2_delta_r_E,S2_delta_r_I,cov_r_E,cov_r_I]
    delta_mu_E,delta_mu_I=sol_M[0:2]
    S2_delta_mu_E,S2_delta_mu_I,cov_mu_E,cov_mu_I=sol_M[2:6]
    sol_M=[mu_E,mu_I,sigma_mu_E**2,sigma_mu_I**2,delta_mu_E,delta_mu_I,
           S2_delta_mu_E,S2_delta_mu_I,cov_mu_E,cov_mu_I]
    return sol_R,sol_M,err
 
##################################################################
##### Compute probability distribution
##################################################################

def P_of_r_A_Base(r_edges,phi,mu,S2_mu,L,Lambda,S_Lambda,frac):
    # F is a function of mu^i+Lambda^i*L
    lambda_i=np.random.normal(Lambda,S_Lambda, N_stat)
    lambda_i[int(frac*N_stat)::]=0;
    lambda_i[lambda_i<0]=0;
    mu_i=np.random.normal(mu,np.sqrt(S2_mu), N_stat)
    r_i=phi(mu_i+lambda_i*L)
    hist_r, r_edges=np.histogram(r_i,r_edges)#,normed=True)
    spacings=np.diff(r_edges);
    hist_r=hist_r/np.sum(hist_r*spacings)
    return hist_r    


def P_of_r_A(r_edges,phi,mu,S2_mu,delta_mu,S2_delta_mu,Cov_mu_delta_mu,L,Lambda,S_Lambda,frac):
    # F is a function of mu^i+Lambda^i*L
    lambda_i=np.random.normal(Lambda,S_Lambda, N_stat)
    lambda_i[int(frac*N_stat)::]=0;
    lambda_i[lambda_i<0]=0;
    if L==0:
        mu_i=np.random.normal(mu,np.sqrt(S2_mu), N_stat)
        r_i=phi(mu_i+lambda_i*L)
    if L>0:
        Mean_Mat=[mu,delta_mu]
        Cov_Mat=np.zeros((2,2))
        Cov_Mat[0,0],Cov_Mat[1,1]=S2_mu,S2_delta_mu
        Cov_Mat[1,0],Cov_Mat[0,1]=Cov_mu_delta_mu,Cov_mu_delta_mu
        mu_i, delta_mu_i = np.random.multivariate_normal(Mean_Mat, Cov_Mat, N_stat).T
        r_i=phi(mu_i+delta_mu_i+lambda_i*L)
    hist_r, r_edges=np.histogram(r_i,r_edges)#,normed=True)
    spacings=np.diff(r_edges);
    hist_r=hist_r/np.sum(hist_r*spacings)
    return hist_r 

def P_of_delta_r_A(delta_r_edges,phi,mu,S2_mu,delta_mu,S2_delta_mu,Cov_mu_delta_mu,L,Lambda,S_Lambda,frac):
    # F is a function of mu^i+Lambda^i*L
    lambda_i=np.random.normal(Lambda,S_Lambda, N_stat)
    lambda_i[int(frac*N_stat)::]=0;
    lambda_i[lambda_i<0]=0;
    Mean_Mat=[mu,delta_mu]
    Cov_Mat=np.zeros((2,2))
    Cov_Mat[0,0],Cov_Mat[1,1]=S2_mu,S2_delta_mu
    Cov_Mat[1,0],Cov_Mat[0,1]=Cov_mu_delta_mu,Cov_mu_delta_mu
    mu_i, delta_mu_i = np.random.multivariate_normal(Mean_Mat, Cov_Mat, N_stat).T
    delta_r_i=phi(mu_i+delta_mu_i+lambda_i*L)-phi(mu_i)
    hist_delta_r, delta_r_edges=np.histogram(delta_r_i,delta_r_edges)#,normed=True)
    spacings=np.diff(delta_r_edges);
    hist_delta_r=hist_delta_r/np.sum(hist_delta_r*spacings)
    return hist_delta_r    


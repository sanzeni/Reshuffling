from numba import jit

import numpy as np
import scipy
from scipy.special import erf, erfi
from scipy.integrate import solve_ivp
from mpmath import fp
from scipy.interpolate import interp1d



    
class Ricciardi(object):
    sr2 = np.sqrt(2)
    sr2π = np.sqrt(2*np.pi)
    srπ = np.sqrt(np.pi)
    
    def __init__(self):
        # Parameters defined by ale
        self.tau_E=0.02;
        self.tau_I=0.01;
        self.theta=20*10**(-3);
        self.V_r=10*10**(-3);
        self.sigma_t=0.01; # Input noise, determine how smooth the single neuron transfer function is
        self.tau_rp=0.002; # Refractory period

        self.ricc_norm=1



    def comp_phi_tab_numerically(self,mu,tau):
        min_u=(self.V_r-mu)/self.sigma_t
        max_u=(self.theta-mu)/self.sigma_t
        r=np.zeros(np.size(mu));
        if np.size(mu)==1:
            if(max_u<10):
                r=1.0/(self.tau_rp+tau*np.sqrt(np.pi)*self.integral(min_u,max_u))
            if(max_u>=10):
                r=max_u/tau/np.sqrt(np.pi)*np.exp(-max_u**2)
        if np.size(mu)>1:
            for idx in range(len(mu)):
                if(max_u[idx]<10):
                    r[idx]=1.0/(self.tau_rp+tau*np.sqrt(np.pi)*self.integral(min_u[idx],max_u[idx]))
                if(max_u[idx]>=10):
                    r[idx]=max_u[idx]/tau/np.sqrt(np.pi)*np.exp(-max_u[idx]**2)
        return r


    def comp_phi_tab(self,mu,tau):
    
        """
        This is the analytical approximation that Ale and Tuan did for numerical efficiency!
        """
        sr2 = np.sqrt(2)
        sr2π = np.sqrt(2*np.pi)
        srπ = np.sqrt(np.pi)
        
        nu_prova=np.linspace(0.,1.0/self.tau_rp,11)
        min_u=(self.V_r-mu)/self.sigma_t
        max_u=(self.theta-mu)/self.sigma_t
        r=np.zeros(np.size(mu));
        if np.size(mu)==1:
            if(min_u>10):
                r=max_u/tau/srπ*np.exp(-max_u**2)
            elif(max_u>-4):
                r=1.0/(self.tau_rp+tau*(0.5*np.pi*\
                                   (erfi(max_u)-erfi(min_u)) +
                                   max_u**2*fp.hyp2f2(1.0,1.0,1.5,2.0,max_u**2) -
                                   min_u**2*fp.hyp2f2(1.0,1.0,1.5,2.0,min_u**2)))
            else:
                r=1.0/(self.tau_rp+tau*(np.log(abs(min_u)) - np.log(abs(max_u)) +
                                   (0.25*min_u**-2 - 0.1875*min_u**-4 + 0.3125*min_u**-6 -
                                    0.8203125*min_u**-8 + 2.953125*min_u**-10) -
                                   (0.25*max_u**-2 - 0.1875*max_u**-4 + 0.3125*max_u**-6 -
                                    0.8203125*max_u**-8 + 2.953125*max_u**-10)))
        if np.size(mu)>1:
            for idx in range(len(mu)):
                if(min_u[idx]>10):
                    r[idx]=max_u[idx]/tau/srπ*np.exp(-max_u[idx]**2)
                elif(min_u[idx]>-4):
                    r[idx]=1.0/(self.tau_rp+tau*(0.5*np.pi*\
                                       (erfi(max_u[idx]) -
                                        erfi(min_u[idx])) +
                                       max_u[idx]**2*fp.hyp2f2(1.0,1.0,1.5,2.0,
                                                                      max_u[idx]**2) -
                                       min_u[idx]**2*fp.hyp2f2(1.0,1.0,1.5,2.0,
                                                                      min_u[idx]**2)))
                else:
                    r[idx]=1.0/(self.tau_rp+tau*(np.log(abs(min_u[idx])) -
                                            np.log(abs(max_u[idx])) +
                                       (0.25*min_u[idx]**-2 - 0.1875*min_u[idx]**-4 +
                                        0.3125*min_u[idx]**-6 - 0.8203125*min_u[idx]**-8 +
                                        2.953125*min_u[idx]**-10) -
                                       (0.25*max_u[idx]**-2 - 0.1875*max_u[idx]**-4 +
                                        0.3125*max_u[idx]**-6 - 0.8203125*max_u[idx]**-8 +
                                        2.953125*max_u[idx]**-10)))
        return r







    def integral(self,mymin,mymax):
        if mymax<11:
            def f(x):
                param=-5.5
                if (x>=param):
                    return np.exp(x**2)*(1+erf(x))
                if (x<param):
                    return -1/np.sqrt(np.pi)/x*(1.0-1.0/2.0*pow(x,-2.0)+3.0/4.0*pow(x,-4.0))
            this_int=integrate.quad(lambda u: f(u),mymin,mymax)
        if mymax>=11:
            this_int=[1./mymax*np.exp(mymax**2)]
        return this_int[0]
        
    def comp_phi_der_tab(self,mu,tau):
        min_u=(self.V_r-mu)/self.sigma_t
        max_u=(self.theta-mu)/self.sigma_t
        r=np.zeros(np.size(mu));
        F1=(np.exp(max_u**2)*(1+erf(max_u)))
        F2=(np.exp(min_u**2)*(1+erf(min_u)))
        return tau*np.sqrt(np.pi)*(F1-F2)/self.sigma_t
        
    def set_up_nonlinearity(self):
    
        mu_tab_max=10.0;
        mu_tab=np.linspace(-mu_tab_max,mu_tab_max,200000)
        mu_tab=np.concatenate(([-10000],mu_tab))
        mu_tab=np.concatenate((mu_tab,[10000]))

        phi_tab_E,phi_tab_I=mu_tab*0,mu_tab*0;
        phi_der_tab_E,phi_der_tab_I=mu_tab*0,mu_tab*0;

        for idx in range(len(phi_tab_E)):
            phi_tab_E[idx]=self.comp_phi_tab(mu_tab[idx],self.tau_E)
            phi_tab_I[idx]=self.comp_phi_tab(mu_tab[idx],self.tau_I)
#            phi_der_tab_E[idx]=self.comp_phi_der_tab(mu_tab[idx],self.tau_E)/phi_tab_E[idx]**2
#            phi_der_tab_I[idx]=self.comp_phi_der_tab(mu_tab[idx],self.tau_I)/phi_tab_I[idx]**2

        self.phi_int_E=interp1d(mu_tab, phi_tab_E, kind='linear')
        self.phi_int_I=interp1d(mu_tab, phi_tab_I, kind='linear')
    
#        self.phi_der_int_E=interp1d(mu_tab, phi_der_tab_E, kind='linear')
#        self.phi_der_int_I=interp1d(mu_tab, phi_der_tab_I, kind='linear')


    
    
    
    

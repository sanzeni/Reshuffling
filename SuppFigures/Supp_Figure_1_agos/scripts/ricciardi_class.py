import pickle
import numpy as np
import torch
import torch_interpolations
import scipy
from scipy.special import erf, erfi
from scipy.integrate import solve_ivp, quad
from mpmath import fp
from scipy.interpolate import interp1d,interpn

def expval(fun,mus,sigs):
    if np.isscalar(mus):
        return quad(lambda z: fun(mus+sigs*z)*np.exp(-z**2/2)/np.sqrt(2*np.pi),-8,8)[0]
    else:
        return [quad(lambda z: fun(mus[i]+sigs[i]*z)*np.exp(-z**2/2)/np.sqrt(2*np.pi),-8,8)[0]
                for i in range(len(mus))]

dmu = 1e-3

def d(fun,mu):
    return (fun(mu+dmu)-fun(mu-dmu))/(2*dmu)

def d2(fun,mu):
    return (fun(mu+dmu)-2*fun(mu)+fun(mu-dmu))/dmu**2
    
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
        
        # nu_prova=np.linspace(0.,1.0/self.tau_rp,11)
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
        
    def set_up_nonlinearity(self,nameout=None):
        save_file = False
        if nameout is not None:
            try:
                with open(nameout+'.pkl', 'rb') as handle:
                    out_dict = pickle.load(handle)
                self.phi_int_E=out_dict['phi_int_E']
                self.phi_int_I=out_dict['phi_int_I']
                print('Loading previously saved nonlinearity')
                return None
            except:
                print('Calculating nonlinearity')
                save_file = True

        mu_tab_max=10.0;
        mu_tab=np.linspace(-mu_tab_max/5,mu_tab_max,int(200000*1.2+1))
        mu_tab=np.concatenate(([-10000],mu_tab))
        mu_tab=np.concatenate((mu_tab,[10000]))

        phi_tab_E,phi_tab_I=mu_tab*0,mu_tab*0;
        # phi_der_tab_E,phi_der_tab_I=mu_tab*0,mu_tab*0;

        for idx in range(len(phi_tab_E)):
            phi_tab_E[idx]=self.comp_phi_tab(mu_tab[idx],self.tau_E)
            phi_tab_I[idx]=self.comp_phi_tab(mu_tab[idx],self.tau_I)
#            phi_der_tab_E[idx]=self.comp_phi_der_tab(mu_tab[idx],self.tau_E)/phi_tab_E[idx]**2
#            phi_der_tab_I[idx]=self.comp_phi_der_tab(mu_tab[idx],self.tau_I)/phi_tab_I[idx]**2

        self.phi_int_E=interp1d(mu_tab, phi_tab_E, kind='linear', fill_value='extrapolate')
        self.phi_int_I=interp1d(mu_tab, phi_tab_I, kind='linear', fill_value='extrapolate')

        if save_file:
            out_dict = {'phi_int_E':self.phi_int_E,
                        'phi_int_I':self.phi_int_I}
            with open(nameout+'.pkl', 'wb') as handle:
                pickle.dump(out_dict,handle)
        
    def set_up_nonlinearity_tensor(self,nameout=None):
        save_file = False
        if nameout is not None:
            try:
                with open(nameout+'.pkl', 'rb') as handle:
                    out_dict = pickle.load(handle)
                self.phi_int_tensor_E=out_dict['phi_int_tensor_E']
                self.phi_int_tensor_I=out_dict['phi_int_tensor_I']
                print('Loading previously saved nonlinearity')
                return None
            except:
                print('Calculating nonlinearity')
                save_file = True

        if hasattr(self,"phi_int_E"):
            u_tab=self.phi_int_E.x
            phi_tab_E=self.phi_int_E.y
            phi_tab_I=self.phi_int_I.y
        else:
            u_tab_max=10.0;
            u_tab=np.linspace(-u_tab_max/5,u_tab_max,int(200000*1.2+1))
            u_tab=np.concatenate(([-10000],u_tab))
            u_tab=np.concatenate((u_tab,[10000]))

            phi_tab_E,phi_tab_I=u_tab*0,u_tab*0;
            # phi_der_tab_E,phi_der_tab_I=u_tab*0,u_tab*0;

            for idx in range(len(phi_tab_E)):
                phi_tab_E[idx]=self.comp_phi_tab(u_tab[idx],self.tau_E)
                phi_tab_I[idx]=self.comp_phi_tab(u_tab[idx],self.tau_I)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using",device)

        u_tab_tensor = torch.from_numpy(u_tab.astype(np.float32)).to(device)
        phi_tab_tensor_E = torch.from_numpy(phi_tab_E.astype(np.float32)).to(device)
        phi_tab_tensor_I = torch.from_numpy(phi_tab_I.astype(np.float32)).to(device)

        self.phi_int_tensor_E=torch_interpolations.RegularGridInterpolator((u_tab_tensor,), phi_tab_tensor_E)
        self.phi_int_tensor_I=torch_interpolations.RegularGridInterpolator((u_tab_tensor,), phi_tab_tensor_I)

        if save_file:
            out_dict = {'phi_int_tensor_E':self.phi_int_tensor_E,
                        'phi_int_tensor_I':self.phi_int_tensor_I}
            with open(nameout+'.pkl', 'wb') as handle:
                pickle.dump(out_dict,handle)

    def set_up_nonlinearity_w_laser(self,LLam,CV_Lam,nameout=None):
        save_file = False
        if nameout is not None:
            try:
                with open(nameout+'.pkl', 'rb') as handle:
                    out_dict = pickle.load(handle)
                self.phiL_int_E=out_dict['phiL_int_E']
                self.phiL2_int_E=out_dict['phiL2_int_E']
                print('Loading previously saved nonlinearity with laser')
                return None
            except:
                print('Calculating nonlinearity with laser')
                save_file = True

        Lvar = np.log(1+CV_Lam**2)
        Lstd = np.sqrt(Lvar)
        Lmean = np.log(LLam)-0.5*Lvar
        
        mu_tab_max=5.0;
        mu_tab=np.linspace(-mu_tab_max/5,mu_tab_max,int(10000*1.2+1))
        mu_tab=np.concatenate(([-1000],mu_tab))
        mu_tab=np.concatenate((mu_tab,[1000]))

        phiL_tab_E=mu_tab*0
        phiL2_tab_E=mu_tab*0

        for idx in range(len(phiL_tab_E)):
            phiL_tab_E[idx]=quad(lambda x: np.exp(-0.5*((np.log(x)-Lmean)/Lstd)**2)/(np.sqrt(2*np.pi)*Lstd*x)*\
                self.phi_int_E(mu_tab[idx]+x),0,50*LLam)[0]
            phiL2_tab_E[idx]=quad(lambda x: np.exp(-0.5*((np.log(x)-Lmean)/Lstd)**2)/(np.sqrt(2*np.pi)*Lstd*x)*\
                self.phi_int_E(mu_tab[idx]+x)**2,0,50*LLam)[0]

        self.phiL_int_E=interp1d(mu_tab, phiL_tab_E, kind='linear', fill_value='extrapolate')
        self.phiL2_int_E=interp1d(mu_tab, phiL2_tab_E, kind='linear', fill_value='extrapolate')

        if save_file:
            out_dict = {'phiL_int_E':self.phiL_int_E,
                        'phiL2_int_E':self.phiL2_int_E}
            with open(nameout+'.pkl', 'wb') as handle:
                pickle.dump(out_dict,handle)
        
    def set_up_mean_nonlinearity(self,nameout=None):
        save_file = False
        if nameout is not None:
            try:
                with open(nameout+'.pkl', 'rb') as handle:
                    out_dict = pickle.load(handle)
                self.mu_tab=out_dict['mu_tab']
                self.sig_tab=out_dict['sig_tab']
                self.M_phi_tab_E=out_dict['M_phi_tab_E']
                self.M_phi_tab_I=out_dict['M_phi_tab_I']
                print('Loading previously saved mean nonlinearity')

                def M_phi_int_E(self,mu,sig):
                    return interpn((self.mu_tab,self.sig_tab), self.M_phi_tab_E, (mu,sig), method='linear', fill_value=None)
                def M_phi_int_I(self,mu,sig):
                    return interpn((self.mu_tab,self.sig_tab), self.M_phi_tab_I, (mu,sig), method='linear', fill_value=None)
                return None
            except:
                print('Calculating mean nonlinearity')
                save_file = True

        mu_tab_max=1.0;
        mu_tab=np.linspace(-mu_tab_max/5,mu_tab_max,int(1000*1.2+1))
        mu_tab=np.concatenate(([-1000],mu_tab))
        mu_tab=np.concatenate((mu_tab,[1000]))

        sig_tab_max=0.1;
        sig_tab=np.linspace(0,sig_tab_max,int(100+1))
        sig_tab=np.concatenate((sig_tab,[1]))

        M_phi_tab_E,M_phi_tab_I=mu_tab[:,None]*sig_tab[None,:]*0,mu_tab[:,None]*sig_tab[None,:]*0;
        # phi_der_tab_E,phi_der_tab_I=mu_tab*0,mu_tab*0;

        for mu_idx in range(len(mu_tab)):
            for sig_idx in range(len(sig_tab)):
                M_phi_tab_E[mu_idx,sig_idx]=expval(self.phi_int_E,mu_tab[mu_idx],sig_tab[sig_idx])
                M_phi_tab_I[mu_idx,sig_idx]=expval(self.phi_int_I,mu_tab[mu_idx],sig_tab[sig_idx])

        self.mu_tab=mu_tab
        self.sig_tab=sig_tab
        self.M_phi_tab_E=M_phi_tab_E
        self.M_phi_tab_I=M_phi_tab_I

        if save_file:
            out_dict = {'mu_tab':self.mu_tab,
                        'sig_tab':self.sig_tab,
                        'M_phi_tab_E':self.M_phi_tab_E,
                        'M_phi_tab_I':self.M_phi_tab_I}
            with open(nameout+'.pkl', 'wb') as handle:
                pickle.dump(out_dict,handle)

    def M_phi_int_E(self,mu,sig):
        return interpn((self.mu_tab,self.sig_tab), self.M_phi_tab_E, (mu,sig), method='linear', fill_value=None)[0]
    def M_phi_int_I(self,mu,sig):
        return interpn((self.mu_tab,self.sig_tab), self.M_phi_tab_I, (mu,sig), method='linear', fill_value=None)[0]
        
    def set_up_mean_nonlinearity_w_laser(self,nameout=None):
        save_file = False
        if nameout is not None:
            try:
                with open(nameout+'.pkl', 'rb') as handle:
                    out_dict = pickle.load(handle)
                self.muL_tab=out_dict['muL_tab']
                self.sigL_tab=out_dict['sigL_tab']
                self.M_phiL_tab_E=out_dict['M_phiL_tab_E']
                self.M_phiL2_tab_E=out_dict['M_phiL2_tab_E']
                print('Loading previously saved mean nonlinearity')

                def M_phi_int_E(self,mu,sig):
                    return interpn((self.mu_tab,self.sig_tab), self.M_phi_tab_E, (mu,sig), method='linear', fill_value=None)
                def M_phi_int_I(self,mu,sig):
                    return interpn((self.mu_tab,self.sig_tab), self.M_phi_tab_I, (mu,sig), method='linear', fill_value=None)
                return None
            except:
                print('Calculating mean nonlinearity')
                save_file = True

        mu_tab_max=1.0;
        mu_tab=np.linspace(-mu_tab_max/5,mu_tab_max,int(1000*1.2+1))
        mu_tab=np.concatenate(([-1000],mu_tab))
        mu_tab=np.concatenate((mu_tab,[1000]))

        sig_tab_max=0.1;
        sig_tab=np.linspace(0,sig_tab_max,int(100+1))
        sig_tab=np.concatenate((sig_tab,[1]))

        M_phiL_tab_E,M_phiL2_tab_E=mu_tab[:,None]*sig_tab[None,:]*0,mu_tab[:,None]*sig_tab[None,:]*0;
        # phi_der_tab_E,phi_der_tab_I=mu_tab*0,mu_tab*0;

        for mu_idx in range(len(mu_tab)):
            for sig_idx in range(len(sig_tab)):
                M_phiL_tab_E[mu_idx,sig_idx]=expval(self.phiL_int_E,mu_tab[mu_idx],sig_tab[sig_idx])
                M_phiL2_tab_E[mu_idx,sig_idx]=expval(self.phiL2_int_E,mu_tab[mu_idx],sig_tab[sig_idx])

        self.muL_tab=mu_tab
        self.sigL_tab=sig_tab
        self.M_phiL_tab_E=M_phiL_tab_E
        self.M_phiL2_tab_E=M_phiL2_tab_E

        if save_file:
            out_dict = {'muL_tab':self.mu_tab,
                        'sigL_tab':self.sig_tab,
                        'M_phiL_tab_E':self.M_phiL_tab_E,
                        'M_phiL2_tab_E':self.M_phiL2_tab_E}
            with open(nameout+'.pkl', 'wb') as handle:
                pickle.dump(out_dict,handle)

    def M_phiL_int_E(self,mu,sig):
        return interpn((self.muL_tab,self.sigL_tab), self.M_phiL_tab_E, (mu,sig), method='linear', fill_value=None)[0]
    def M_phiL2_int_E(self,mu,sig):
        return interpn((self.muL_tab,self.sigL_tab), self.M_phiL2_tab_E, (mu,sig), method='linear', fill_value=None)[0]

    def phiE_tensor(self,u):
        # return self.comp_phi_tensor(u,self.tau_E)
        return self.phi_int_tensor_E(u[None,:])
    def phiI_tensor(self,u):
        # return self.comp_phi_tensor(u,self.tau_I)
        return self.phi_int_tensor_I(u[None,:])

    def phiE(self,u):
        return self.phi_int_E(u)
    def phiI(self,u):
        return self.phi_int_I(u)
    def phiLE(self,u):
        return self.phiL_int_E(u)
    
    

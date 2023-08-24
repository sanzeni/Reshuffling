import numpy as np
import numpy.matlib
import scipy
from scipy.special import erf, erfi
from scipy.integrate import solve_ivp
from scipy.linalg import circulant
from mpmath import fp

#test xcodeproj
    
class network(object):
    """

    Parameters
    ----------
    seed_con : int
        Random seed
    n : int
        Number of cell types
    NE : int
        Number of excitatory neurons per location
    ori_map : 2darray
        Orientation preference map
    dl : float
        Length of the entire 2D grid
    Sl : 1darray
        2-vector with spatial connection probability widths
    Sori : 1darray
        2-vector with orientation connection probability widths
        
    """
    
    
    
    def __init__(self, seed_con=0, n=2, Nl=25, NE=8, gamma=0.25, dl=1, pmax=0.09, Sl=None, Sori=None, Stun=None, ori_type='columnar'):
        self.seed_con = seed_con

        # Set external seed
        np.random.seed(seed_con)

        self.Nl = Nl
        self.NE = NE
        self.gamma = gamma
        self.NI = round(self.NE*self.gamma)
        self.NT = self.NE+self.NI
        self.Nloc = self.Nl**2

        self.Einds = [slice(loc*self.NT,loc*self.NT+self.NE) for loc in range(self.Nloc)]
        self.Iinds = [slice(loc*self.NT+self.NE,(loc+1)*self.NT) for loc in range(self.Nloc)]
        self.allE = np.zeros(0,np.int8)
        self.allI = np.zeros(0,np.int8)
        for loc in range(self.Nloc):
            Elocinds = np.arange(self.Einds[loc].start,self.Einds[loc].stop)
            Ilocinds = np.arange(self.Iinds[loc].start,self.Iinds[loc].stop)
            self.allE = np.append(self.allE,Elocinds)
            self.allI = np.append(self.allI,Ilocinds)

        self.ori_type = ori_type
        self.ori_map = self.generate_orientation_map(3,ori_type)
        
        # Total number of neurons
        self.N = self.ori_map.size

        self.n = n

        self.dl = dl
        self.dx = dl / self.Nl
        self.Sori = Sori
        self.Sl = Sl
        self.Stun = Stun

        self.SpaceDim = 2
        self.Dim = 3
        self.set_XYZ()

        self.spatial_profile = 'nonlamnormgaussian'
        self.normalize_by_mean = True

        # Hardcoded network params
        self.GE,self.GI = 1.0,2.0 # Gain of Excitatory and inhibitory cells and I cells
        self.w_EE = 1;

    def set_seed(self,seed_con):
        np.random.seed(seed_con)


    def generate_orientation_map(self,sf,ori_type):
        """
        Generate the network's orientation preference map

        Parameters
        ----------
        sf : int or float
            Spatial frequency
        columnar : bool
            Whether the orientation map is columnarly organized

        Returns
        -------
        2darray
            Matrix of orientation preferences
        """

        n = 500; # This parameter doesnt matter so much


        if hasattr(self, 'ori_map'):
            print('You already had an orientation map, but now you are re-writting it!')
            
        #sf This one sets the spatial freq, we will have that amount of max in a side
        kc = (2*np.pi)*(sf);

        [X,Y] = np.meshgrid(np.linspace(0,1,self.Nl),np.linspace(0,1,self.Nl));

        z = np.zeros_like(X);
        for j in range(n):

            kj = kc * np.array([ np.cos(j*np.pi/n) , np.sin(j*np.pi/n) ]);
            sj = 2*np.random.randint(1,2)-3;
            phij = np.random.rand()*2*np.pi;
            tmp = (X*kj[0]+Y*kj[1])*sj + phij;
            z = z + np.exp(1j * tmp);

        OMap = np.angle(z);
        OMap = OMap-np.min(OMap);
        OMap = OMap/np.max(OMap);
        OMap = OMap*180;

        if ori_type=='columnar':
            OMap = OMap
        if ori_type=='saltandpepper':
            OMap = np.random.random_sample(OMap.shape)*(np.max(OMap))
        if ori_type=='vanilla':
            OMap = OMap*0

        OMapOut = np.tile(OMap,(self.NT,1,1)).transpose(1, 2, 0)
        return OMapOut
        
        
    def get_center_orientation(self):
        return self.ori_map[int(self.Nl/2),int(self.Nl/2),0]
        
    def get_neurons_tuning(self):
        return self.ori_map.flatten()

    def get_oriented_neurons(self,delta_ori=15,grating_orientation=None):
        Tuning_vec = self.get_neurons_tuning()
        if grating_orientation==None:
            grating_orientation = self.get_center_orientation()
        lb = grating_orientation-delta_ori
        ub = grating_orientation+delta_ori

        if 0 <= lb and ub <= 180:    # both within
            this_neurons, = np.where(np.logical_and(lb<Tuning_vec, Tuning_vec<ub))
            
        elif 0 > lb and ub <= 180:  # lb is negative
            true_lb = np.mod(lb,180)
            this_neurons, = np.where(np.logical_or(true_lb<Tuning_vec, Tuning_vec<ub))
            
        elif 0 <= lb and 180 < ub: # ub is bigger than 180
            true_ub = np.mod(ub,180)
            this_neurons, = np.where(np.logical_or(lb<Tuning_vec, Tuning_vec<true_ub))
            
        elif 0 > lb and 180 < ub: # all oris
            true_lb = np.mod(lb,180)
            true_ub = np.mod(ub,180)
            this_neurons, = np.where(np.logical_and(true_lb<Tuning_vec, Tuning_vec<true_ub))
        else:
            print(lb,ub)
        return this_neurons

    def get_centered_neurons(self,stim_size=0.5,grating_center=None):
        if grating_center==None:
            grating_center = (np.ones(self.SpaceDim)*self.Nl/2).astype(int)
        Delta_vec = np.abs(self.XY - grating_center)*self.dl/self.Nl
        Delta_vec[:,0] = self.make_periodic(Delta_vec[:,0],self.dl/2)
        Delta_vec[:,1] = self.make_periodic(Delta_vec[:,1],self.dl/2)
        Distance_vec = np.sqrt(Delta_vec[:,0]**2 + Delta_vec[:,1]**2)

        this_neurons, = np.where(Distance_vec<stim_size)
        return this_neurons

    def get_neurons_at_given_ori_distance_to_grating(self,diff_width=5, degs_away_from_center=15, grating_orientation=None, signed=False):

        Tuning_vec = self.get_neurons_tuning()
        if grating_orientation is None:

            grating_orientation = self.get_center_orientation()
        if signed:
            Tuning_vec = self.make_periodic(Tuning_vec - grating_orientation,90)
        else:
            Tuning_vec = self.make_periodic(np.abs(Tuning_vec - grating_orientation),90)

        lb = degs_away_from_center-diff_width
        ub = degs_away_from_center+diff_width


        if 0 <= lb and ub <= 180:    # both within
            this_neurons=np.logical_and(lb<Tuning_vec, Tuning_vec<ub)

        elif 0 > lb and ub <= 180:  # lb is negative
            true_lb = np.mod(lb,180)
            this_neurons=np.logical_or(true_lb<Tuning_vec, Tuning_vec<ub)

        elif 0 <= lb and 180 < ub: # ub is bigger than 180
            true_ub = np.mod(ub,180)
            this_neurons=np.logical_or(lb<Tuning_vec, Tuning_vec<true_ub)

        elif 0 > lb and 180 < ub: # all oris
            true_lb = np.mod(lb,180)
            true_ub = np.mod(ub,180)
            this_neurons=np.logical_and(true_lb<Tuning_vec, Tuning_vec<true_ub)

        return this_neurons

        # lb_m = grating_orientation-degs_away_from_center-diff_width
        # ub_m = grating_orientation-degs_away_from_center+diff_width

        # lb_p = grating_orientation+degs_away_from_center-diff_width
        # ub_p = grating_orientation+degs_away_from_center+diff_width


        # if 0 <= lb_m and ub_m <= 180:    # both within
        #     this_neurons_m=np.logical_and(lb_m<Tuning_vec, Tuning_vec<ub_m)

        # elif 0 > lb_m and ub_m <= 180:  # lb is negative
        #     true_lb = np.mod(lb_m,180)
        #     this_neurons_m=np.logical_or(true_lb<Tuning_vec, Tuning_vec<ub_m)

        # elif 0 <= lb_m and 180 < ub_m: # ub is bigger than 180
        #     true_ub = np.mod(ub_m,180)
        #     this_neurons_m=np.logical_or(lb_m<Tuning_vec, Tuning_vec<true_ub)

        # elif 0 > lb_m and 180 < ub_m: # all oris
        #     true_lb = np.mod(lb_m,180)
        #     true_ub = np.mod(ub_m,180)
        #     this_neurons_m=np.logical_and(true_lb<Tuning_vec, Tuning_vec<true_ub)

        # if 0 <= lb_p and ub_p <= 180:    # both within
        #     this_neurons_p=np.logical_and(lb_p<Tuning_vec, Tuning_vec<ub_p)

        # elif 0 > lb_p and ub_p <= 180:  # lb is negative
        #     true_lb = np.mod(lb_p,180)
        #     this_neurons_p=np.logical_or(true_lb<Tuning_vec, Tuning_vec<ub_p)

        # elif 0 <= lb_p and 180 < ub_p: # ub is bigger than 180
        #     true_ub = np.mod(ub_p,180)
        #     this_neurons_p=np.logical_or(lb_p<Tuning_vec, Tuning_vec<true_ub)

        # elif 0 > lb_p and 180 < ub_p: # all oris
        #     true_lb = np.mod(lb_p,180)
        #     true_ub = np.mod(ub_p,180)
        #     this_neurons_p=np.logical_and(true_lb<Tuning_vec, Tuning_vec<true_ub)

        # if signed:
        #     return this_neurons_p
        # else:
        #     this_neurons=np.logical_or(this_neurons_p,this_neurons_m)
        #     return this_neurons
    ###############################################################
    # Basic functions
    
    def sub2ind(self,array_shape, rows, cols, heights=None):
        if len(array_shape)==2:
            ind = rows*array_shape[1] + cols
            ind[ind < 0] = -1
            ind[ind >= array_shape[0]*array_shape[1]] = -1
            return ind

        elif len(array_shape)==3:
            if heights is None:
                heights = np.zeros_like(rows)
            ind = rows*array_shape[1] + cols
            ind = ind*array_shape[2] + heights
            ind[ind < 0] = -1
            ind[ind >= array_shape[0]*array_shape[1]*array_shape[2]] = -1
            return ind

    def ind2sub(self,array_shape, ind):
        if len(array_shape)==1:
            return ind
        
        elif len(array_shape)==2:
            rows = (ind/ array_shape[1]).astype('int')
            cols = ind % array_shape[1]
            return np.array([rows, cols])
        
        elif len(array_shape)==3:
            ind = (ind/array_shape[2]).astype('int') # This is because every N neurons have the same position
            rows = (ind/ array_shape[1]).astype('int')
            cols = ind % array_shape[1]
            return np.array([rows, cols])
        
    def make_periodic(self,vec_in,half_period):
        vec_out = np.copy(vec_in);
        vec_out[vec_out >  half_period] =  2*half_period-vec_out[vec_out >  half_period];
        vec_out[vec_out < -half_period] = -2*half_period-vec_out[vec_out < -half_period];
        return vec_out
        

    ###############################################################
    # Generate Gaussian and Orientation dependent inputs


    def set_XYZ(self):
    
        self.XY = np.transpose(self.ind2sub(self.ori_map.shape,np.arange(0,self.ori_map.size,1)))
        self.Z = self.ori_map.flatten()
        if self.Dim==1:
            self.XY = np.array([self.XY])
            self.Z = np.array([self.Z])

    def make_orientation_difference_input(self,Orientation):
        PreferredOri = self.Z;
        dOri = self.make_periodic(abs(PreferredOri - Orientation),90)

        InO = np.exp(-(dOri**2)/(2*self.Stun**2));
        # InO = np.exp(np.cos(dOri*2*np.pi/180)/(self.Stun*2*np.pi/180)**2) / np.exp(1/(self.Stun*2*np.pi/180)**2)
        
        return InO.flatten()


    def make_patch_input(self,Stim_Cont,Stim_Size,patch_type='nonlamnormgaussian',xc=None,RF_size=None):
        sigma_input = Stim_Size*self.dl
        if xc is None:
            xc = (np.ones(self.SpaceDim)*self.Nl/2).astype(int)
        if RF_size is None:
            RF_size = np.ones(self.n)*0.01

        centered_XY = self.XY-np.matlib.repmat(np.transpose(xc),self.XY.shape[0],1)
        for ndims in range(self.SpaceDim):
            centered_XY[:,ndims] = self.make_periodic(np.abs(centered_XY[:,ndims]),self.Nl/2) ;
        rspace = np.sqrt(np.sum(centered_XY**2,axis = -1))*self.dx

        InSr = np.zeros_like(rspace)
        for npops in range(self.n):
            if npops == 0:
                inds = self.allE
            else:
                inds = self.allI

            if patch_type=='gaussian':
                InSr[inds] = Stim_Cont[npops]*np.exp(-rspace[inds]**2/(2*sigma_input**2))*self.dx**2/2/np.pi/sigma_input**2;

            if patch_type=='nonlamnormgaussian':
                InSr[inds] = Stim_Cont[npops]*np.exp(-rspace[inds]**2/(2*sigma_input**2))#*self.dx**2/2/np.pi/sigma_input**2;
                
            if patch_type=='constant' or Stim_Size==np.inf:
                InSr[inds] = Stim_Cont[npops]*np.ones_like(rspace[inds])
                
            if patch_type=='circle':
                InSr[inds] =Stim_Cont[npops]*( 1-1/(1+np.exp(-(rspace[inds] - sigma_input/2)/RF_size[npops])));

            elif patch_type=='stict_circle':
                InSr[inds] = rspace[inds]<sigma_input
                    
        return InSr



    def make_grating(self,Stim_Cont,Stim_Size,Stim_Ori,patch_type='nonlamnormgaussian',xc=None,RF_size=None):
        circle = self.make_patch_input(Stim_Cont,Stim_Size,patch_type,xc,RF_size)
        ori = self.make_orientation_difference_input(Stim_Ori)
        grating = circle*ori
        return grating


    ###############################################################
    ##Indexes (source of all evil, Taiga Abe 2019)
            
    def grid2neuronindex(self,x0,y0,z0):
        NN0 = z0+y0*self.NT+x0*self.NT*self.Nl   #This is the position in the connectivity matrix of this connection
        return NN0


    def neuronindex2grid(self,NN0):
        z = np.mod(NN0,self.NT)
        y = np.mod(int(np.floor(((NN0-z))/self.NT)),self.Nl)
        x = int(np.floor((NN0-y*self.NT-z)/(self.NT*self.Nl)))
        return x,y,z


    def flat2grid(self,A):
        return A.reshape((self.Nl,self.Nl,self.NT))[:,:,0].flatten()



    ####################################################################################################################
    ##Connectivity Matrix
        
        
    def generate_single_circulant(self,lam):
        
        MyNloc = int(self.Nl/2+1)
        v = np.zeros(MyNloc)
        vtotal = np.zeros(self.Nl)
        for k in range(MyNloc):
        
            if self.spatial_profile=='gaussian':
                v[k]= np.exp(-(k*self.dx)**2/(2*lam**2))/np.sqrt(2*np.pi)/lam;

            if self.spatial_profile=='nonlamnormgaussian':
                v[k]= np.exp(-(k*self.dx)**2/(2*lam**2))
                
            elif self.spatial_profile=='exponential':
                v[k] = np.exp(-np.abs(k*self.dx)/lam)
                
            elif self.spatial_profile=='nonspatial':
                v[k] = 1
        
        if self.Nl%2 == 1:
            vtotal = np.append(v,np.flip(v[1:],0))
        else:
            vtotal = np.append(v,np.flip(v[1:-1],0))
        C1D = circulant(vtotal)
        C2D = np.kron(C1D,C1D)
            
        return C2D


    def generate_full_circulant(self):
        Cfull = np.zeros((self.N,self.N))
        for i in range(self.n):
            if i == 0:
                i_inds = self.Einds
                i_all = self.allE
                Ni = self.NE
            else:
                i_inds = self.Iinds
                i_all = self.allI
                Ni = self.NI
            for j in range(self.n):
                if j == 0:
                    j_inds = self.Einds
                    j_all = self.allE
                    Nj = self.NE
                else:
                    j_inds = self.Iinds
                    j_all = self.allI
                    Nj = self.NI
                for loc in range(self.Nloc):
                    Cfull[i_inds[loc],j_all] =\
                        np.kron(self.generate_single_circulant(self.Sl[i,j]),np.ones((Ni,j_all.size)))
        return Cfull


    # This function below gives the exact same matrix than the one above,
    #     but is easier to generalize to non circulant matrices
    def generate_single_circulant_By_Hand(self,lam):

        SqdeltaD = np.zeros((self.Nloc,self.Nloc))
        for n in range(self.n):
            XdMat = np.matlib.repmat(self.XY[:,n],self.Nloc,1)
            XDiffMat_nonnomr = np.abs(XdMat-np.transpose(XdMat))
            XDist = self.make_periodic(XDiffMat_nonnomr,self.Nl/2)
            SqdeltaD += (XDist)**2
            
        deltaD = np.sqrt(SqdeltaD)*self.dx

        if self.spatial_profile=='gaussian':
            C= np.exp(-deltaD**2/(2*lam**2))/np.sqrt(2*np.pi)/lam;

        if self.spatial_profile=='nonlamnormgaussian':
            C= np.exp(-deltaD**2/(2*lam**2))
            
        elif self.spatial_profile=='exponential':
            C = np.exp(-np.abs(deltaD)/lam)
            
        elif self.spatial_profile=='nonspatial':
            C = np.ones_like(deltaD)
            
        return C
    
    

    def generate_full_connectivity(self,Wmat,VarMat,pmax,CV_K,vanilla_or_not,givemeMeanVariance=True):

        ################################
        C_full = np.zeros((self.N,self.N))
        W_mean_full = np.zeros((self.N,self.N))
        W_var_full = np.zeros((self.N,self.N))
        
#         if vanilla_or_not!='vanilla':

#             print('Your Sl is = ' + str(self.Sl))
#             print('Your Sori is = ' + str(self.Sori))

        for i in range(self.n):
            if i == 0:
                i_inds = self.Einds
                i_all = self.allE
                Ni = self.NE
            else:
                i_inds = self.Iinds
                i_all = self.allI
                Ni = self.NI
            for j in range(self.n):
                if j == 0:
                    j_inds = self.Einds
                    j_all = self.allE
                    Nj = self.NE
                else:
                    j_inds = self.Iinds
                    j_all = self.allI
                    Nj = self.NI
                if vanilla_or_not=='vanilla' or vanilla_or_not==True:
                    W_space = np.ones((self.Nloc,self.Nloc))
                    W_Ori = np.ones((self.Nloc,self.Nloc))
                else:
                    W_space_aux = self.generate_single_circulant(self.Sl[i,j])
                    W_Ori_aux   = self.generate_single_orientation_diff_connectivity(self.Sori[i,j])
                    if self.normalize_by_mean:
                        W_space = W_space_aux/np.mean(W_space_aux)
                        W_Ori   = W_Ori_aux/np.mean(W_Ori_aux)
                    else:
                        W_space = W_space_aux
                        W_Ori   = W_Ori_aux
 
                #################################### HERE TUAN #######################################
                # calculate connection probability from each location to each other location for this connection type
                ps = np.maximum(pmax * Nj * W_space * W_Ori,1e-12)
                # calculate mean in-degree for each location for this connection type
                Ks = np.sum(pmax * Nj * W_space * W_Ori,1)
                # for each location, assign presynaptic connections with total in-degree varying according to CV_K
                for loc in range(self.Nloc):
                    K = Ks[loc]
                    random_K=np.rint(np.random.normal(K, CV_K*K,Ni)).astype('int')
                    random_K[random_K<0] = 0
                    if i==j: random_K[random_K>Nj*self.Nloc-1] = Nj*self.Nloc-1
                    else: random_K[random_K>Nj*self.Nloc] = Nj*self.Nloc
                    possible_idx_post = range(i_inds[loc].start,i_inds[loc].stop)
                    possible_idx_pre = j_all;
                    prob_idx_pre = np.kron(ps[loc,:],np.ones(Nj))
                    count = 0
                    for idx_post in possible_idx_post:
                        # prevent autapses
                        mask = (possible_idx_pre!=idx_post)
                        array_idx_pre=np.random.choice(possible_idx_pre[mask],replace=False,
                            p=prob_idx_pre[mask]/np.sum(prob_idx_pre[mask]),size=random_K[count])
                        C_full[idx_post,array_idx_pre]=1
                        count += 1
                    W_mean_full[i_inds[loc],j_all] = Wmat[i,j]
                    W_var_full[i_inds[loc],j_all] = VarMat[i,j]
                #################################### HERE TUAN #######################################

        
        if givemeMeanVariance:
            return C_full, W_mean_full,W_var_full
        else:
            return C_full

    def generate_connection_strength_disorder(self):
        Wvar = np.zeros((2,2))
        return Wvar

    def get_disorder_input(self,J,gE,gI,beta,rX,KX,Stim_Size,vanilla_or_not,patch_type='nonlamnormgaussian'):
        w_IE=self.w_EE/beta;
        w_EI=gE*self.w_EE;
        w_II=gI*w_IE;
        w_EX,w_IX=(self.GI*self.gamma*gE-self.GE)*self.w_EE,(self.GI*self.gamma*gI-self.GE)*w_IE;
        wX=np.asarray([w_EX,w_IX]);
        
        if vanilla_or_not=='vanilla' or vanilla_or_not==True:
            self.H = np.zeros(self.N)
            self.H[self.allE] = rX*wX[0]
            self.H[self.allI] = rX*wX[1]
            self.H *= J*KX
        else:
            grating = self.make_grating(rX*wX,Stim_Size,self.get_center_orientation(),patch_type)
            self.H = J*KX*grating
            

    def generate_disorder(self,J,gE,gI,beta,pmax,CV_K,rX,KX,Lam,CV_Lam,Stim_Size,vanilla_or_not,patch_type='nonlamnormgaussian'):

        w_IE=self.w_EE/beta;
        w_EI=gE*self.w_EE;
        w_II=gI*w_IE;
        w_EX,w_IX=(self.GI*self.gamma*gE-self.GE)*self.w_EE,(self.GI*self.gamma*gI-self.GE)*w_IE;
        wX=np.asarray([w_EX,w_IX]);
        w=np.zeros((2,2));
        w[0,:]=self.w_EE,-w_EI
        w[1,:]=w_IE,-w_II

    
        Wvar=self.generate_connection_strength_disorder()
        
        C_full, W_mean_full,W_var_full = self.generate_full_connectivity(J*w,J*Wvar,pmax,CV_K,vanilla_or_not,True)
        self.M = C_full*(W_mean_full+np.random.normal(size=(self.N,self.N))*np.sqrt(W_var_full))
        
        self.get_disorder_input(J,gE,gI,beta,rX,KX,Stim_Size,vanilla_or_not,patch_type)

        sigma_l = np.sqrt(np.log(1+CV_Lam**2))
        mu_l = np.log(Lam)-sigma_l**2/2
        self.LAM = np.zeros(self.N)
        self.LAM[self.allE] = np.random.lognormal(mu_l, sigma_l, self.NE*self.Nloc)
        

    def generate_single_orientation_diff_connectivity(self,lam):

        ZdMat = np.matlib.repmat(self.flat2grid(self.Z),self.Nloc,1)
        ZDiffMat_nonnomr = np.abs(ZdMat-np.transpose(ZdMat))
        deltaZ = self.make_periodic(ZDiffMat_nonnomr,90)
        
        if self.spatial_profile=='gaussian':
            OC = np.exp(-deltaZ**2/(2*lam**2))/np.sqrt(2*np.pi)/lam;
        elif self.spatial_profile=='nonnormgaussian':
            OC = np.exp(-deltaZ**2/(2*lam**2))/np.sqrt(2*np.pi)/lam
        elif self.spatial_profile=='nonlamnormgaussian':
            OC = np.exp(-deltaZ**2/(2*lam**2))#/np.sqrt(2*np.pi)/lam
            
        elif self.spatial_profile=='nonspatial':
            OC = np.ones_like(deltaZ)
        elif self.spatial_profile=='exponential':
            OC = np.exp(-np.abs(deltaZ)/lam)
        elif self.mouse_monkey=='Vanilla':
            OC = np.ones_like(deltaZ)
        return OC


    def generate_full_orientation_diff_connectivity(self):
        Cfull = np.zeros((self.N,self.N))
        for i in range(self.n):
            if i == 0:
                i_inds = self.Einds
                i_all = self.allE
                Ni = self.NE
            else:
                i_inds = self.Iinds
                i_all = self.allI
                Ni = self.NI
            for j in range(self.n):
                if j == 0:
                    j_inds = self.Einds
                    j_all = self.allE
                    Nj = self.NE
                else:
                    j_inds = self.Iinds
                    j_all = self.allI
                    Nj = self.NI
                for loc in range(self.Nloc):
                    Cfull[i_inds[loc],j_all] =\
                        np.kron(self.generate_single_orientation_diff_connectivity(self.Sori[i,j]),
                            np.ones((Ni,j_all.size)))
        return Cfull
        
        


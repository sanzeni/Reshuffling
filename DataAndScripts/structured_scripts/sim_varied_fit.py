#!/usr/bin/python
import sys
import os
import argparse

import numpy as np
import pickle

import data_analysis as da
import validate_utils as vu
import load_utils as lu

import ricciardi_class as ric
import network as network

try:
    data=da.Data_MonkeyMouse('both','./../../../DataAndScripts/experimental_data/')
except:
    data=da.Data_MonkeyMouse('both','./../../DataAndScripts/experimental_data/')

ri = ric.Ricciardi()
ri.set_up_nonlinearity()

parser = argparse.ArgumentParser(description=('This python script takes the best fit structured model '
    'and simulates the network with varied parameters for Supp Figure 5'))
parser.add_argument('-p', '--panel',    help='which panel to compute',type=str, default='cmb_mous')
args = vars(parser.parse_args())
panel = args['panel']

try:
    with open('./../../../DataAndScripts/structured_scripts'+"/Model_Fit_Separately_StructModel-"+"Best"+'.pkl', 'rb') as handle:
        comb_fit = pickle.load(handle)
except:
    with open('./../../DataAndScripts/structured_scripts'+"/Model_Fit_Separately_StructModel-"+"Best"+'.pkl', 'rb') as handle:
        comb_fit = pickle.load(handle)

    
fit_preds = {}
fit_params = {}
fit_rX = {}

for anim_idx in range(2):
    fit_preds[anim_idx] = comb_fit['predictions_of_r_sim_'+data.this_animals[anim_idx]]
    fit_params[anim_idx] = np.concatenate(([0,500,0.09,30,1e-3,200],
                                           comb_fit['best_params_'+data.this_animals[anim_idx]]))
    fit_rX[anim_idx] = comb_fit['best_rXs_'+data.this_animals[anim_idx]]

seeds=[1,3,5,7]
max_min=45

seed_con = fit_params[1][vu.res_param_idxs_fixed['seed_con']]
KX       = fit_params[1][vu.res_param_idxs_fixed['KX']]
pmax     = fit_params[1][vu.res_param_idxs_fixed['pmax']]
SoriE    = fit_params[1][vu.res_param_idxs_fixed['SoriE']]
Lam      = fit_params[1][vu.res_param_idxs_fixed['Lam']]
Tmax_over_tau_E =fit_params[1][vu.res_param_idxs_fixed['Tmax_over_tau_E']]

T = np.arange(0,1.5*Tmax_over_tau_E*ri.tau_E,ri.tau_I/3);
mask_time=T>(0.5*Tmax_over_tau_E*ri.tau_E)

J         = fit_params[1][vu.res_param_idxs['J']]
GI        = fit_params[1][vu.res_param_idxs['GI']]
gE        = fit_params[1][vu.res_param_idxs['gE']]
gI        = fit_params[1][vu.res_param_idxs['gI']]
beta      = fit_params[1][vu.res_param_idxs['beta']]
rX        = fit_rX[1][-1]

CV_K      = fit_params[1][vu.res_param_idxs['CV_K']]
SlE       = fit_params[1][vu.res_param_idxs['SlE']]
SlI       = fit_params[1][vu.res_param_idxs['SlI']]
SoriI     = fit_params[1][vu.res_param_idxs['SoriI']]
Stun      = fit_params[1][vu.res_param_idxs['Stun']]
CV_Lam    = fit_params[1][vu.res_param_idxs['CV_Lam']]
L         = fit_params[1][vu.res_param_idxs['L']]

stim_size=0.5
ori_type='columnar'
RF='in'
tuned='yes'

net = network.network(seed_con=int(seed_con), n=2, Nl=25, NE=8, gamma=0.25, dl=1,
                      Sl=np.array([[SlE,SlI],[SlE,SlI]]), Sori=np.array([[SoriE,SoriI],[SoriE,SoriI]]),
                      Stun=Stun, ori_type=ori_type)
net.GI = GI

print("Parameters used seed= {:d} // GI= {:.2f} // gE= {:.2f} // gI= {:.2f} // beta= {:.2f} // KX= {:d} // pmax= {:.2f}" \
.format(int(seed_con),GI,gE,gI,beta,int(KX),pmax))
print("CV_K= {:.4f} // SlE= {:.3f} // SlI= {:.3f} // SoriE= {:.2f} // SoriI= {:.2f} // Stun= {:.2f}"\
.format(CV_K,SlE,SlI,SoriE,SoriI,Stun))
print("Lam= {:.3f} // CV_Lam= {:.2f} // J= {:.6f} // rX= {:.2f} // L= {:.2f} // Tmax_over_tau_E= {:d}"\
.format(Lam,CV_Lam,J,rX,L,int(Tmax_over_tau_E)))
print('')

def sim_scan_two_prms(preds,param1,param2,varparams1,varparams2,mask=None):
    if mask is None:
        mask = np.ones((len(varparams1),len(varparams2))) > 0
    
    params_dict={}
    params_dict['Stim_Size']=0.5
    params_dict['seed_con']=int(seed_con)
    params_dict['KX']=int(KX)
    params_dict['pmax']=pmax
    params_dict['SoriE']=SoriE
    params_dict['Lam']=Lam
    params_dict['J']=J
    params_dict['GI']=GI
    params_dict['gE']=gE
    params_dict['gI']=gI
    params_dict['beta']=beta
    params_dict['CV_K']=CV_K
    params_dict['SlE']=SlE
    params_dict['SlI']=SlI
    params_dict['SoriI']=SoriI
    params_dict['Stun']=Stun
    params_dict['CV_Lam']=CV_Lam
    params_dict['L']=L

    params_dict['Nl']=25
    params_dict['NE']=8
    params_dict['n']=2
    params_dict['gamma']=0.25
    params_dict['dl']=1

    params_dict['ori_type']=ori_type
    params_dict['vanilla_or_not']=False
    
    params_dict['rX']=rX
    
    for idx1 in range(len(varparams1)):
        for idx2 in range(len(varparams2)):
            print((idx1,idx2))
            
            this_params_dict = params_dict.copy()
            this_params_dict[param1] = varparams1[idx1]
            this_params_dict[param2] = varparams2[idx2]
            
            if mask[idx1,idx2]:
                _,preds_tuned,preds_untuned,_,_,_ =\
                    lu.sim_const_map(this_params_dict,[this_params_dict['rX']],ri,T,mask_time,RF,tuned,
                                     seed_con,seeds,max_min)

                preds[idx1,idx2] = preds_tuned[0]
            else:
                preds[idx1,idx2] = np.nan

if panel=='A':
    JAs = J*10**np.linspace(-2,0,10)[[4,6,8,9]]
    rXAs = rX*10**np.linspace(-2,0,10)
    predAs = np.zeros((len(JAs),len(rXAs),6))
    sim_scan_two_prms(predAs,'J','rX',JAs,rXAs)
    line_data = JAs
    x_data = rXAs
    preds = predAs
    delr_data = predAs[:,:,1]-predAs[:,:,0]
    sig_delr_data = predAs[:,:,4]
    rho_data = predAs[:,:,5]
elif panel=='B':
    rXBs = rX*10**np.linspace(-2,0,10)[[4,6,8,9]]
    JBs = J*10**np.linspace(-2,0,10)
    predBs = np.zeros((len(rXBs),len(JBs),6))
    sim_scan_two_prms(predBs,'rX','J',rXBs,JBs)
    line_data = rXBs
    x_data = JBs
    preds = predBs
    delr_data = predBs[:,:,1]-predBs[:,:,0]
    sig_delr_data = predBs[:,:,4]
    rho_data = predBs[:,:,5]
elif panel=='C':
    gICs = np.arange(2.0,3.5+0.1,0.5)
    gECs = np.arange(2.0,5.5+0.1,0.5)
    predCs = np.zeros((len(gICs),len(gECs),6))
    sim_scan_two_prms(predCs,'gI','gE',gICs,gECs,gECs >= gICs[:,None])
    line_data = gICs
    x_data = gECs
    preds = predCs
    delr_data = predCs[:,:,1]-predCs[:,:,0]
    sig_delr_data = predCs[:,:,4]
    rho_data = predCs[:,:,5]
elif panel=='D':
    JDs = J*10**np.linspace(-2,0,10)[[4,6,8,9]]
    CV_LamDs = CV_Lam*10**np.linspace(-2,0,10)
    predDs = np.zeros((len(JDs),len(CV_LamDs),6))
    sim_scan_two_prms(predDs,'J','CV_Lam',JDs,CV_LamDs)
    line_data = JDs
    x_data = CV_LamDs
    preds = predDs
    delr_data = predDs[:,:,1]-predDs[:,:,0]
    sig_delr_data = predDs[:,:,4]
    rho_data = predDs[:,:,5]
elif panel=='E':
    StunEs = np.arange(15,45+1,10)
    SoriIEs = np.arange(15,45+1,5)
    predEs = np.zeros((len(StunEs),len(SoriIEs),6))
    sim_scan_two_prms(predEs,'Stun','SoriI',StunEs,SoriIEs)
    line_data = StunEs
    x_data = SoriIEs
    preds = predEs
    delr_data = predEs[:,:,1]-predEs[:,:,0]
    sig_delr_data = predEs[:,:,4]
    rho_data = predEs[:,:,5]

scan_dict={}

scan_dict['line_data']      = line_data
scan_dict['x_data']         = x_data
scan_dict['preds']          = preds
scan_dict['delr_data']      = delr_data
scan_dict['sig_delr_data']  = sig_delr_data
scan_dict['rho_data']       = rho_data

output_dir='./'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_dir+"Scan_Figure_"+panel+'.pkl', 'wb') as handle:
    pickle.dump(scan_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

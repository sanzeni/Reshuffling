from numba import jit
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
from scipy.special import erf, erfi
from scipy.integrate import solve_ivp
from mpmath import fp
import pickle
import pickle5
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib

#import network as network
import data_analysis as da

#import ricciardi_class as ricciardi_class
#ri=ricciardi_class.Ricciardi()
########################################################################################################################
########################################################################################################################
########################################################################################################################

# Define plotting style
font = {'family' : 'normal', 'weight' : 'normal', 'size' : 7, 'family' : 'serif', 'serif' : ['Arial']}
mpl.rc('font', **font)
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42


##### Load data
try:
    data=da.Data_MonkeyMouse('both','./../../DataAndScripts/experimental_data')
except:
    data=da.Data_MonkeyMouse('both','./../../../DataAndScripts/experimental_data')


def plot_all_normalization_curves(output_norm_data,output_norm_unstr,output_norm_str,name_end_out):


    for which_set, which_set_name in zip([output_norm_data,output_norm_unstr,  output_norm_str],['data','unstr','str']):

        for animal, animal_idx in zip(data.this_animals,range(2)):

            for redux in [True, False]:

                #########################
                # Structured
                name_out='Animal='+animal+'_Reduced='+str(redux)
                if which_set_name=='str':
                    name_file_out='Normalization_Curves_Model='+which_set_name+'_'+name_out+name_end_out
                else:
                    name_file_out='Normalization_Curves_Model='+which_set_name+'_'+name_out

                title=which_set_name+' Animal='+animal+' Reduced='+str(redux)
                plot_normalization_curves_for_each(which_set['contrast_'+name_out],\
                                                 which_set['Rates_Matrix_flatten_'+name_out],\
                                                 which_set['all_fits_'+name_out], \
                                                 which_set['Rsq_'+name_out], \
                                             title, name_file_out)


def plot_normalization_curves_for_each(contrast,Rates_Matrix_flatten, all_fits, Rqs, title, name_out):

    nc=len(contrast)
    fig, axs = plt.subplots(1,3, figsize=(10,1.8), dpi=300, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .3, wspace=.3)

    cmap_list = matplotlib.cm.get_cmap('Spectral')

    axs[0].set_prop_cycle('color',[cmap_list(k) for k in np.linspace(0, 1, Rates_Matrix_flatten.shape[0])])
    axs[0].plot(contrast*100, Rates_Matrix_flatten[:,:nc].T, 'o', contrast*100,all_fits[:,:nc].T, '-',);
    axs[0].set_xlabel('Contrast')
    axs[0].set_ylabel('Firing Rate (Hz)')
    axs[0].set_title(title +' Baseline')
    axs[0].set_ylim([0,1.2*np.max(Rates_Matrix_flatten[:,:nc])])
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(4))
    axs[0].yaxis.set_major_locator(plt.MaxNLocator(4))


    axs[1].set_prop_cycle('color',[cmap_list(k) for k in np.linspace(0, 1, Rates_Matrix_flatten.shape[0])])
    axs[1].plot(contrast*100, Rates_Matrix_flatten[:,nc:].T, 'o', contrast*100,all_fits[:,nc:].T, '-',);
    axs[1].set_xlabel('Contrast')
    axs[1].set_ylabel('Firing Rate (Hz)')
    axs[1].set_title(title + ' w/ Opto')
    axs[1].set_ylim([0,1.2*np.max(Rates_Matrix_flatten[:,:nc])])
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(3))
    axs[1].yaxis.set_major_locator(plt.MaxNLocator(3))

    axs[1].sharex(axs[0])
    axs[1].sharey(axs[0])

    axs[2].hist(Rqs,density=True);
    axs[2].set_xlabel(r'$R^2$')
    axs[2].set_ylabel('PDF')
    axs[2].set_title(r'$R^2$')
    axs[2].set_xlim([0,1.01])
    axs[2].xaxis.set_major_locator(plt.MaxNLocator(3))
    axs[2].yaxis.set_major_locator(plt.MaxNLocator(3))

    nameout='./figs/'+ name_out
    fig.savefig(nameout+'.pdf', bbox_inches='tight')
    



def plot_Normalization_Histograms_Models_and_Data(output_norm_data,output_norm_unstr,output_norm_str,name_end_out):

    # Define plotting style
    font = {'family' : 'normal', 'weight' : 'normal', 'size' : 7, 'family' : 'serif', 'serif' : ['Arial']}
    mpl.rc('font', **font)
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['mathtext.rm'] = 'serif'
    mpl.rcParams['pdf.fonttype'] = 42



    pars_names=['opto','Rm','Ro','n','m','D','sig','S']
    pars_names_rdx=['Rm','Ro','n','D','sig','S']


    
    
    for animal, animal_idx in zip(data.this_animals,range(2)):

        name_out='Animal='+animal+'_Reduced=False'

        ncells=output_norm_data['all_pars_'+name_out].shape[0]
        ncells_str=output_norm_str['all_pars_'+name_out].shape[0]
        ncells_unstr=output_norm_unstr['all_pars_'+name_out].shape[0]

        chosen_cells_str=np.random.randint(ncells_str, size=ncells)
        chosen_cells_unstr=np.random.randint(ncells_unstr, size=ncells)
        
        for redux in [True, False]:
            if redux:
                pars_names=['Rm','Ro','n','D','sig','S']
            else:
                pars_names=['opto','Rm','Ro','n','m','D','sig','S']

            fig, axs = plt.subplots(3,len(pars_names), figsize=(len(pars_names),2.2), dpi=200, facecolor='w', edgecolor='k',sharex=False, sharey='col')
            fig.subplots_adjust(hspace =0.1, wspace=.05)

            name_out='Animal='+animal+'_Reduced='+str(redux)
            name_file_out='Normalization_Histograms_Comparative_'+name_out
            pars_nr=len(output_norm_data['all_pars_'+name_out][0])

            fig.suptitle('Animal='+animal+' Reduced='+str(redux))
            for k in range(pars_nr):
                lb=np.percentile(output_norm_data['all_pars_'+name_out][:,k],2)
                ub=np.percentile(output_norm_data['all_pars_'+name_out][:,k],95)
                bins=np.linspace(lb,ub,15)
                

                axs[0,k].hist(output_norm_data['all_pars_'+name_out][:,k],bins,density=True)
                axs[0,k].set_xticklabels([])
                axs[0,k].tick_params(axis='both', which='major', pad=-2)


                axs[1,k].hist(output_norm_str['all_pars_'+name_out][chosen_cells_str,k],bins,density=True,color='orange')
                axs[1,k].set_xticklabels([])
                axs[1,k].tick_params(axis='both', which='major', pad=-2)

                axs[2,k].hist(output_norm_unstr['all_pars_'+name_out][chosen_cells_unstr,k],bins,density=True,color='red')
                axs[2,k].set_xlabel(pars_names[k], labelpad=1)
                axs[2,k].tick_params(axis='both', which='major', pad=-2)


            fig.tight_layout()
            nameout='./figs/'+name_file_out+name_end_out

            fig.savefig(nameout+'.pdf', bbox_inches='tight')


def plot_Normalization_D_Over_Models_and_Data(output_norm_data,output_norm_unstr,output_norm_str,name_end_out):

    for animal, animal_idx in zip(data.this_animals,range(2)):

        for redux in [True, False]:
            if redux:
                pars_names=['Rm','Ro','n','D','sig','S']
            else:
                pars_names=['opto','Rm','Ro','n','m','D','sig','S']

            fig, axs = plt.subplots(1,1, figsize=(3,3), dpi=400, facecolor='w', edgecolor='k',sharex=False, sharey=False)
            fig.subplots_adjust(hspace =0.1, wspace=.1)

            name_out='Animal='+animal+'_Reduced='+str(redux)
            name_file_out='Normalization_D_Over_S_Animal='+animal+'_Reduced='+str(redux)
            pars_nr=len(output_norm_data['all_pars_'+name_out][0])

            fig.suptitle('Animal='+animal+' Reduced='+str(redux))
            lb=0.01#np.percentile(output_norm_data['all_pars_'+name_out][:,-3]/output_norm_data['all_pars_'+name_out][:,-1],2)
            ub=1.5#np.percentile(output_norm_data['all_pars_'+name_out][:,-3]/output_norm_data['all_pars_'+name_out][:,-1],95)
            bins=np.linspace(lb,ub,15)
            axs.hist(output_norm_data['all_pars_'+name_out][:,-3]/output_norm_data['all_pars_'+name_out][:,-1],bins,density=True,histtype='step', color='blue',)
            axs.hist(output_norm_str['all_pars_'+name_out][:,-3]/output_norm_str['all_pars_'+name_out][:,-1],bins,density=True,histtype='step',color='orange',)
            axs.hist(output_norm_unstr['all_pars_'+name_out][:,-3]/output_norm_unstr['all_pars_'+name_out][:,-1],bins,density=True,histtype='step',color='red')
            axs.set_xlabel('D/S')
            fig.tight_layout()
            
            nameout='./figs/'+name_file_out+name_end_out
        
            fig.savefig(nameout+'.pdf', bbox_inches='tight')


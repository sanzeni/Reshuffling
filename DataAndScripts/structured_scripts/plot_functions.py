import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
from scipy.special import erf, erfi
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from mpmath import fp
try:
    import pickle5 as pickle
except:
    import pickle
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib

import network as network
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




def plot_fitted_data(output_dir,name_fit,all_moments=None):
    ##### Load data
    monkey_mouse_data=da.Data_MonkeyMouse('both','./../../../DataAndScripts/experimental_data')

    #For this plots the structure needs to be  (contrast x opto x neurons)
    try:
        with open(output_dir+'/'+name_fit+'.pkl', 'rb') as handle_loadModel:
            this_fit=pickle.load(handle_loadModel)
    except:
        with open(output_dir+'/'+name_fit+'.pkl', 'rb') as handle_loadModel:
            this_fit=pickle5.load(handle_loadModel)
    try:
        namesims='Simulation_'+name_fit
        print(namesims)
        try:
            with open(output_dir+'/'+namesims+'.pkl', 'rb') as handle_loadModel:
                output_sims=pickle.load(handle_loadModel)
                print(output_sims.keys())
        except:
            with open(output_dir+'/'+namesims+'.pkl', 'rb') as handle_loadModel:
                output_sims=pickle5.load(handle_loadModel)
                print(output_sims.keys())
        with_sims=True
        nameout=output_dir+'/'+name_fit+'with_sims'
    except:
        print('No sims with name ' + namesims)
        with_sims=False
        nameout=output_dir+'/'+name_fit

    if with_sims:
        rows=6
    else:
        rows=4
    cols=2
    colors=['c','m']
    color_sim=['xkcd:aqua','xkcd:coral']
    fig, axs = plt.subplots(rows,cols, figsize=(2.5*cols,2.*rows), dpi=300, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.5)
    axs = axs.ravel()
    this_animals=['mouse','monkey']
    
    for idx in range(len(monkey_mouse_data.this_animals)):
        dataset=monkey_mouse_data.bootstrap_moments[idx]
        contrast=monkey_mouse_data.contrast[idx]
        predictions=this_fit['predictions_'+monkey_mouse_data.this_animals[idx]]
        if with_sims:
            transposed = (output_sims['predictions_of_r_sim_mouse'].shape[0] == 7)
            animal_preds=output_sims['predictions_of_r_sim_'+monkey_mouse_data.this_animals[idx]]
        for idx_moment in range(6):
            if idx_moment<2:
                idx_row=0
                ccc=['k',colors[idx]][idx_moment]
                ccd=['k',color_sim[idx]][idx_moment]
                ymin=0
                ymax=[25,80][idx]
                ylabel='Mean rate (spk/s)'

            if (idx_moment>=2)&(idx_moment<4):
                idx_row=1
                ccc=['k',colors[idx]][idx_moment-2]
                ccd=['k',color_sim[idx]][idx_moment-2]
                ymin=0
                ymax=[20,60][idx]
                ylabel='Std rate (spk/s)'

            if (idx_moment==4):
                idx_row=2
                ccc=colors[idx]
                ccd=color_sim[idx]
                ymin=0
                ymax=[20,60][idx]
                ylabel='Std Δrate (spk/s)'

            if (idx_moment==5):
                idx_row=3
                ccc=colors[idx]
                ccd=color_sim[idx]
                ymin=-0.75
                ymax=0.25
                ylabel='ρ'
            
            m,sem=dataset[idx_moment,:,0],dataset[idx_moment,:,1]
            axs[2*idx_row+idx].scatter(contrast,m,facecolors='none', s=10,edgecolors=ccc,marker='o')
            axs[2*idx_row+idx].fill_between(contrast,m-sem,m+sem,color=ccc,alpha=0.2)
            if with_sims:
                if transposed:
                    axs[2*idx_row+idx].plot(contrast,animal_preds[:,idx_moment],'--',color=ccd,alpha=1.)
                else:
                    axs[2*idx_row+idx].plot(contrast,animal_preds[idx_moment,:],'--',color=ccd,alpha=1.)
            axs[2*idx_row+idx].plot(contrast,predictions[idx_moment,:],color=ccc,alpha=1.)
#            if all_moments not None:

            axs[2*idx_row+idx].set_xscale('symlog', linthresh=12)
            axs[2*idx_row+idx].set_ylim([ymin,ymax])
            axs[2*idx_row+idx].set_xlabel('contrast (%)')
            axs[2*idx_row+idx].set_ylabel(ylabel)
        axs[2*idx_row+idx].axhline(y=0,ls='--',c='k')

    if with_sims:
        for idx in range(len(monkey_mouse_data.this_animals)):
            contrast=monkey_mouse_data.contrast[idx]
            try:
                animal_balance=output_sims['full_balance_'+monkey_mouse_data.this_animals[idx]]
                animal_optorat=output_sims['full_opto_ratio_'+monkey_mouse_data.this_animals[idx]]
                full = True
            except:
                animal_balance=output_sims['balance_'+monkey_mouse_data.this_animals[idx]]
                animal_optorat=output_sims['opto_ratio_'+monkey_mouse_data.this_animals[idx]]
                full = False
            for idx_row in range(4,5+1):
                ccc=colors[idx]
                ccd=color_sim[idx]
                if idx_row==4:
                    ylabel='Balance index β'
                    y=animal_balance
                elif idx_row==5:
                    ylabel='Opto Input Ratio'
                    y=animal_optorat
                if full:
                    axs[2*idx_row+idx].plot(contrast,y[:,0],'-.',color=ccd,alpha=1.)
                    axs[2*idx_row+idx].plot(contrast,y[:,1],':',color=ccd,alpha=1.)
                else:
                    axs[2*idx_row+idx].plot(contrast,y,'--',color=ccd,alpha=1.)
                axs[2*idx_row+idx].set_xscale('symlog', linthresh=12)
                axs[2*idx_row+idx].set_yscale('log')
                # axs[2*idx_row+idx].set_ylim([ymin,ymax])
                axs[2*idx_row+idx].set_xlabel('contrast (%)')
                axs[2*idx_row+idx].set_ylabel(ylabel)


    print('Done')

    fig.tight_layout()
    fig.savefig(nameout+'.pdf', bbox_inches='tight')


def plot_fitted_data_from_dict(output_sims,output_dir,name_fit,all_moments=None):
    ##### Load data
    monkey_mouse_data=da.Data_MonkeyMouse('both','./../../../DataAndScripts/experimental_data')

    with_sims=False
    nameout=output_dir+'/'+name_fit

    rows=6
    cols=2
    colors=['c','m']
    color_sim=['xkcd:aqua','xkcd:coral']
    fig, axs = plt.subplots(rows,cols, figsize=(2.5*cols,2.*rows), dpi=300, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.5)
    axs = axs.ravel()
    this_animals=['mouse','monkey']

    transposed = (output_sims['predictions_of_r_sim_mouse'].shape[0] == 7)
    
    for idx in range(len(monkey_mouse_data.this_animals)):
        dataset=monkey_mouse_data.bootstrap_moments[idx]
        contrast=monkey_mouse_data.contrast[idx]
        predictions=output_sims['predictions_of_r_sim_'+monkey_mouse_data.this_animals[idx]]
        for idx_moment in range(6):
            if idx_moment<2:
                idx_row=0
                ccc=['k',colors[idx]][idx_moment]
                ccd=['k',color_sim[idx]][idx_moment]
                ymin=0
                ymax=[25,80][idx]
                ylabel='Mean rate (spk/s)'

            if (idx_moment>=2)&(idx_moment<4):
                idx_row=1
                ccc=['k',colors[idx]][idx_moment-2]
                ccd=['k',color_sim[idx]][idx_moment-2]
                ymin=0
                ymax=[20,60][idx]
                ylabel='Std rate (spk/s)'

            if (idx_moment==4):
                idx_row=2
                ccc=colors[idx]
                ccd=color_sim[idx]
                ymin=0
                ymax=[20,60][idx]
                ylabel='Std Δrate (spk/s)'

            if (idx_moment==5):
                idx_row=3
                ccc=colors[idx]
                ccd=color_sim[idx]
                ymin=-0.75
                ymax=0.25
                ylabel='ρ'
            
            m,sem=dataset[idx_moment,:,0],dataset[idx_moment,:,1]
            axs[2*idx_row+idx].scatter(contrast,m,facecolors='none', s=10,edgecolors=ccc,marker='o')
            axs[2*idx_row+idx].fill_between(contrast,m-sem,m+sem,color=ccc,alpha=0.2)
            if transposed:
                axs[2*idx_row+idx].plot(contrast,predictions[:,idx_moment],color=ccc,alpha=1.)
            else:
                axs[2*idx_row+idx].plot(contrast,predictions[idx_moment,:],color=ccc,alpha=1.)
#            if all_moments not None:

            axs[2*idx_row+idx].set_xscale('symlog', linthresh=12)
            axs[2*idx_row+idx].set_ylim([ymin,ymax])
            axs[2*idx_row+idx].set_xlabel('contrast (%)')
            axs[2*idx_row+idx].set_ylabel(ylabel)
        axs[2*idx_row+idx].axhline(y=0,ls='--',c='k')

    for idx in range(len(monkey_mouse_data.this_animals)):
        contrast=monkey_mouse_data.contrast[idx]
        try:
            animal_balance=output_sims['full_balance_'+monkey_mouse_data.this_animals[idx]]
            animal_optorat=output_sims['full_opto_ratio_'+monkey_mouse_data.this_animals[idx]]
            full = True
        except:
            animal_balance=output_sims['balance_'+monkey_mouse_data.this_animals[idx]]
            animal_optorat=output_sims['opto_ratio_'+monkey_mouse_data.this_animals[idx]]
            full = False
        for idx_row in range(4,5+1):
            ccc=colors[idx]
            ccd=color_sim[idx]
            if idx_row==4:
                ylabel='Balance index β'
                y=animal_balance
            elif idx_row==5:
                ylabel='Opto Input Ratio'
                y=animal_optorat
            if full:
                axs[2*idx_row+idx].plot(contrast,y[:,0],'-.',color=ccc,alpha=1.)
                axs[2*idx_row+idx].plot(contrast,y[:,1],':',color=ccc,alpha=1.)
            else:
                axs[2*idx_row+idx].plot(contrast,y,color=ccc,alpha=1.)
            axs[2*idx_row+idx].set_xscale('symlog', linthresh=12)
            axs[2*idx_row+idx].set_yscale('log')
            # axs[2*idx_row+idx].set_ylim([ymin,ymax])
            axs[2*idx_row+idx].set_xlabel('contrast (%)')
            axs[2*idx_row+idx].set_ylabel(ylabel)


    print('Done')

    fig.tight_layout()
    fig.savefig(nameout+'.pdf', bbox_inches='tight')




def plot_best_preds_from_validate_param(best_preds,with_contrast=False):

    output_dir='./../plot_validate_with_GI'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
        
    data=da.Data_MonkeyMouse('both','./../../../DataAndScripts/experimental_data')
    dataset = data.bootstrap_moments
    contrast = data.contrast



    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cols = 2
    rows = 4
    colors=['c','m']
    color_sim=['xkcd:aqua','xkcd:coral']
    fig, axs = plt.subplots(rows,cols, figsize=(2.5*cols,2.*rows), dpi=300, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.5)
    axs = axs.ravel()



    for idx in range(len(data.this_animals)):
        dataset=data.bootstrap_moments[idx]
        contrast=data.contrast[idx]

        for idx_moment in range(6):
            if idx_moment<2:
                idx_row=0
                ccc=['k',colors[idx]][idx_moment]
                ccd=['k',color_sim[idx]][idx_moment]
                ymin=0
                ymax=[25,80][idx]
                ylabel='Mean rate (spk/s)'

            if (idx_moment>=2)&(idx_moment<4):
                idx_row=1
                ccc=['k',colors[idx]][idx_moment-2]
                ccd=['k',color_sim[idx]][idx_moment-2]
                ymin=0
                ymax=[20,60][idx]
                ylabel='Std rate (spk/s)'

            if (idx_moment==4):
                idx_row=2
                ccc=colors[idx]
                ccd=color_sim[idx]
                ymin=0
                ymax=[20,60][idx]
                ylabel='Std Δrate (spk/s)'

            if (idx_moment==5):
                idx_row=3
                ccc=colors[idx]
                ccd=color_sim[idx]
                ymin=-0.75
                ymax=0.25
                ylabel='ρ'


            for i in range(data.nc[idx]):
                if not with_contrast:
                    for preds in best_preds[idx][i]:
                        
                        try:
                            axs[2*idx_row+idx].plot(contrast[i],preds[:,idx_moment],"s-",color=ccc,alpha=0.3)
                        except:
                            try:
                                axs[2*idx_row+idx].plot(contrast[i],preds[idx_moment],"s-",color=ccc,alpha=0.3)
                            except:
                                pass
                elif with_contrast:
                    for preds_wc in best_preds[idx][i]:

                        try:
                            axs[2*idx_row+idx].plot(contrast,preds_wc[:,idx_moment,:].T,"s-",color=ccc,alpha=0.3)
                        except:
                            try:
                                axs[2*idx_row+idx].plot(contrast,preds_wc[idx_moment,:],"s-",color=ccc,alpha=0.3)
                            except:
                                pass


            m,sem=dataset[idx_moment,:,0],dataset[idx_moment,:,1]
            axs[2*idx_row+idx].scatter(contrast,m,facecolors='none', s=10,edgecolors=ccc,marker='o')
            axs[2*idx_row+idx].fill_between(contrast,m-sem,m+sem,color=ccc,alpha=0.2)


            axs[2*idx_row+idx].set_xscale('symlog', linthresh=12)
            axs[2*idx_row+idx].set_ylim([ymin,ymax])
            axs[2*idx_row+idx].set_xlabel('contrast (%)')
            axs[2*idx_row+idx].set_ylabel(ylabel)
        axs[2*idx_row+idx].axhline(y=0,ls='--',c='k')


    print('Done')
    if with_contrast:
        nameout=output_dir+'/Best_preds_from_validate_param_w_contrast'
    else:
        nameout=output_dir+'/Best_preds_from_validate_param'


    fig.tight_layout()
    fig.savefig(nameout+'.pdf', bbox_inches='tight')


def plot_preds(preds,output_dir,name_end,animal='both',preds_aux=None,plot_delta_r=False):
    data=da.Data_MonkeyMouse('both','../data',calc_delta_r=plot_delta_r)

    if animal=='both':
        animal_idxs=[0,1]
    elif animal in ('mouse','Mouse'):
        animal_idxs=[0]
    elif animal in ('monkey','Monkey'):
        animal_idxs=[1]

    if plot_delta_r:
        rows=5
        n_moment=7
    else:
        rows=4
        n_moment=6
    cols=len(animal_idxs)
    colors=['c','m']
    color_sim=['xkcd:aqua','xkcd:coral']
    fig, axs = plt.subplots(rows,cols, figsize=(2.5*cols,2.*rows), dpi=300, facecolor='w', edgecolor='k', sharex=True)
    fig.subplots_adjust(hspace = .5, wspace=.5)
    axs = axs.ravel()
    this_animals=['mouse','monkey']
    
    for idx,anim_idx in enumerate(animal_idxs):
        dataset=data.bootstrap_moments[anim_idx]
        contrast=data.contrast[anim_idx]
        if preds[0].ndim==2:
            predictions=preds[anim_idx]
            if preds_aux is not None:
                predictions_aux=preds_aux[anim_idx]
        else:
            predictions=preds
            if preds_aux is not None:
                predictions_aux=preds_aux
        for idx_moment in range(n_moment):
            if idx_moment<2:
                idx_row=0
                ccc=['k',colors[anim_idx]][idx_moment]
                ccd=['k',color_sim[anim_idx]][idx_moment]
                ymin=0
                ymax=[25,80][anim_idx]
                if preds_aux is not None:
                    ymax=max(ymax,*predictions_aux[:,[0,1]].flatten())
                ylabel='Mean rate (spk/s)'

            if (idx_moment>=2)&(idx_moment<4):
                idx_row=1
                ccc=['k',colors[anim_idx]][idx_moment-2]
                ccd=['k',color_sim[anim_idx]][idx_moment-2]
                ymin=0
                ymax=[20,60][anim_idx]
                if preds_aux is not None:
                    ymax=max(ymax,*predictions_aux[:,[2,3]].flatten())
                ylabel='Std rate (spk/s)'

            if (idx_moment==4):
                idx_row=2
                ccc=colors[anim_idx]
                ccd=color_sim[anim_idx]
                ymin=0
                ymax=[20,60][anim_idx]
                ylabel='Std Δrate (spk/s)'

            if (idx_moment==5):
                idx_row=3
                ccc=colors[anim_idx]
                ccd=color_sim[anim_idx]
                ymin=-0.75
                ymax=0.25
                ylabel='ρ'

            if (idx_moment==6):
                idx_row=4
                ccc=colors[anim_idx]
                ccd=color_sim[anim_idx]
                ymin=[-5,-10][anim_idx]
                ymax=[20,30][anim_idx]
                ylabel='Mean Δrate (spk/s)'
            
            m,sem=dataset[idx_moment,:,0],dataset[idx_moment,:,1]
            axs[cols*idx_row+idx].scatter(contrast,m,facecolors='none', s=10,edgecolors=ccc,marker='o')
            axs[cols*idx_row+idx].fill_between(contrast,m-sem,m+sem,color=ccc,alpha=0.2)
            axs[cols*idx_row+idx].plot(contrast,predictions[:,idx_moment],color=ccc,alpha=1.)
            if preds_aux is not None:
                axs[cols*idx_row+idx].plot(contrast,predictions_aux[:,idx_moment],color=ccc,ls='--',alpha=1.)
#            if all_moments not None:
            if idx_moment >= 5:
                axs[cols*idx_row+idx].axhline(y=0,ls='--',c='k')

            axs[cols*idx_row+idx].set_xscale('symlog', linthresh=12)
            axs[cols*idx_row+idx].set_ylim([ymin,ymax])
            axs[cols*idx_row+idx].set_xlabel('contrast (%)')
            axs[cols*idx_row+idx].set_ylabel(ylabel)


    print('Done')

    fig.tight_layout()
    fig.savefig(output_dir+'/preds_'+name_end+'.pdf', bbox_inches='tight')


def plot_preds_with_bal(preds,bals,optrs,output_dir,name_end,animal='both',preds_aux=None,bals_aux=None,optrs_aux=None,plot_delta_r=False):
    data=da.Data_MonkeyMouse('both','../data',calc_delta_r=plot_delta_r)

    if animal=='both':
        animal_idxs=[0,1]
    elif animal in ('mouse','Mouse'):
        animal_idxs=[0]
    elif animal in ('monkey','Monkey'):
        animal_idxs=[1]

    if plot_delta_r:
        rows=7
        n_moment=7
    else:
        rows=6
        n_moment=6
    cols=len(animal_idxs)
    colors=['c','m']
    color_sim=['xkcd:aqua','xkcd:coral']
    fig, axs = plt.subplots(rows,cols, figsize=(2.5*cols,2.*rows), dpi=300, facecolor='w', edgecolor='k', sharex=True)
    fig.subplots_adjust(hspace = .5, wspace=.5)
    axs = axs.ravel()
    this_animals=['mouse','monkey']
    
    for idx,anim_idx in enumerate(animal_idxs):
        dataset=data.bootstrap_moments[anim_idx]
        contrast=data.contrast[anim_idx]
        if preds[0].ndim==2:
            predictions=preds[anim_idx]
            if preds_aux is not None:
                predictions_aux=preds_aux[anim_idx]
        else:
            predictions=preds
            if preds_aux is not None:
                predictions_aux=preds_aux
        for idx_moment in range(n_moment):
            if idx_moment<2:
                idx_row=0
                ccc=['k',colors[anim_idx]][idx_moment]
                ccd=['k',color_sim[anim_idx]][idx_moment]
                ymin=0
                ymax=[25,80][anim_idx]
                if preds_aux is not None:
                    ymax=max(ymax,*predictions_aux[:,[0,1]].flatten())
                ylabel='Mean rate (spk/s)'

            if (idx_moment>=2)&(idx_moment<4):
                idx_row=1
                ccc=['k',colors[anim_idx]][idx_moment-2]
                ccd=['k',color_sim[anim_idx]][idx_moment-2]
                ymin=0
                ymax=[20,60][anim_idx]
                if preds_aux is not None:
                    ymax=max(ymax,*predictions_aux[:,[2,3]].flatten())
                ylabel='Std rate (spk/s)'

            if (idx_moment==4):
                idx_row=2
                ccc=colors[anim_idx]
                ccd=color_sim[anim_idx]
                ymin=0
                ymax=[20,60][anim_idx]
                ylabel='Std Δrate (spk/s)'

            if (idx_moment==5):
                idx_row=3
                ccc=colors[anim_idx]
                ccd=color_sim[anim_idx]
                ymin=-0.75
                ymax=0.25
                ylabel='ρ'

            if (idx_moment==6):
                idx_row=4
                ccc=colors[anim_idx]
                ccd=color_sim[anim_idx]
                ymin=[-5,-10][anim_idx]
                ymax=[20,30][anim_idx]
                ylabel='Mean Δrate (spk/s)'
            
            m,sem=dataset[idx_moment,:,0],dataset[idx_moment,:,1]
            axs[cols*idx_row+idx].scatter(contrast,m,facecolors='none', s=10,edgecolors=ccc,marker='o')
            axs[cols*idx_row+idx].fill_between(contrast,m-sem,m+sem,color=ccc,alpha=0.2)
            axs[cols*idx_row+idx].plot(contrast,predictions[:,idx_moment],color=ccc,alpha=1.)
            if preds_aux is not None:
                axs[cols*idx_row+idx].plot(contrast,predictions_aux[:,idx_moment],color=ccc,ls='--',alpha=1.)
#            if all_moments not None:
            if idx_moment >= 5:
                axs[cols*idx_row+idx].axhline(y=0,ls='--',c='k')

            axs[cols*idx_row+idx].set_xscale('symlog', linthresh=12)
            axs[cols*idx_row+idx].set_ylim([ymin,ymax])
            axs[cols*idx_row+idx].set_xlabel('contrast (%)')
            axs[cols*idx_row+idx].set_ylabel(ylabel)
    
    for idx,anim_idx in enumerate(animal_idxs):
        contrast=data.contrast[anim_idx]
        if preds[0].ndim==2:
            animal_balance=bals[anim_idx]
            animal_optorat=optrs[anim_idx]
            if bals_aux:
                animal_balance_aux=bals_aux[anim_idx]
            if optrs_aux:
                animal_optorat_aux=optrs_aux[anim_idx]
        else:
            animal_balance=bals
            animal_optorat=optrs
            if bals_aux is not None:
                animal_balance_aux=bals_aux
            if optrs_aux is not None:
                animal_optorat_aux=optrs_aux
        for idx_row in range(rows-2,rows):
            ccc=colors[anim_idx]
            ccd=color_sim[anim_idx]
            if idx_row==rows-2:
                ylabel='Balance index β'
                y=animal_balance
                if bals_aux is not None:
                    y_aux=animal_balance_aux
                ymin=1e-2
                ymax=1e0
            elif idx_row==rows-1:
                ylabel='Opto Input Ratio'
                y=animal_optorat
                if optrs_aux is not None:
                    y_aux=animal_optorat_aux
                ymin=2e-3
                ymax=2e-1
            try:
                axs[cols*idx_row+idx].plot(contrast,y[:,anim_idx],color=ccc,alpha=1.)
                if (idx_row==4 and bals_aux is not None) or (idx_row==5 and optrs_aux is not None):
                    axs[cols*idx_row+idx].plot(contrast,y_aux[:,anim_idx],color=ccc,ls='--',alpha=1.)
            except:
                axs[cols*idx_row+idx].plot(contrast,y,color=ccc,alpha=1.)
                if (idx_row==4 and bals_aux is not None) or (idx_row==5 and optrs_aux is not None):
                    axs[cols*idx_row+idx].plot(contrast,y_aux,color=ccc,ls='--',alpha=1.)
            # axs[cols*idx_row+idx].plot(contrast,y[:,0],'-',color=ccc,alpha=1.)
            # axs[cols*idx_row+idx].plot(contrast,y[:,1],'--',color=ccc,alpha=1.)
            axs[cols*idx_row+idx].set_xscale('symlog', linthresh=12)
            axs[cols*idx_row+idx].set_yscale('log')
            axs[cols*idx_row+idx].set_ylim([ymin,ymax])
            axs[cols*idx_row+idx].set_xlabel('contrast (%)')
            axs[cols*idx_row+idx].set_ylabel(ylabel)


    print('Done')

    fig.tight_layout()
    fig.savefig(output_dir+'/preds_with_bal_'+name_end+'.pdf', bbox_inches='tight')


def plot_bal(bals,optrs,output_dir,name_end):
    data=da.Data_MonkeyMouse('both','./../../../DataAndScripts/experimental_data')

    rows=2
    cols=2
    colors=['c','m']
    color_sim=['xkcd:aqua','xkcd:coral']
    fig, axs = plt.subplots(rows,cols, figsize=(2.5*cols,2.*rows), dpi=300, facecolor='w', edgecolor='k', sharex=True)
    fig.subplots_adjust(hspace = .5, wspace=.5)
    axs = axs.ravel()
    this_animals=['mouse','monkey']

    for idx in range(len(data.this_animals)):
        contrast=data.contrast[idx]
        animal_balance=bals[idx]
        animal_optorat=optrs[idx]
        for idx_row in range(0,1+1):
            ccc=colors[idx]
            ccd=color_sim[idx]
            if idx_row==0:
                ylabel=r'Balance index $\beta$'
                y=animal_balance
                ymin=4e-2
                ymax=4e-1
                color='k'
            elif idx_row==1:
                ylabel='Optogenetic/excitatory input'
                y=animal_optorat
                ymin=4e-3
                ymax=4e-1
                color=ccc
            axs[2*idx_row+idx].plot(contrast,y[:,0],'-.',color=color,alpha=1.)
            axs[2*idx_row+idx].plot(contrast,y[:,1],':',color=color,alpha=1.)
            axs[2*idx_row+idx].set_xscale('symlog', linthresh=12)
            axs[2*idx_row+idx].set_yscale('log')
            axs[2*idx_row+idx].set_ylim([ymin,ymax])
            axs[2*idx_row+idx].set_xlabel('contrast (%)')
            axs[2*idx_row+idx].set_ylabel(ylabel)


    print('Done')

    fig.tight_layout()
    fig.savefig(output_dir+'/bal_'+name_end+'.pdf', bbox_inches='tight')


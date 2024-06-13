#Script for plotting trf decoder weights; can do grand average across all subjects, within a single category, or single subject

#%%
# INIT
import pickle
import scipy.io as spio
import numpy as np
import os
import h5py
from scipy import signal
import utils
from mtrf.model import TRF
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
from trf_helpers import get_subj_trf_pth, get_thresh_folds_dir



#%%
# topoplotting script

if __name__=='__main__':
    
    # set colorbar upper and lower bounds
    topo_lims=(-0.06, 0.06)

    grand_avg=True
    
    results_dir=os.path.join("..","results")
    # Single cat only matters if grand_avg=True (then do grand average of single category)
    single_cat="hc" #set to None or mute this line to get all subjects
    clean_or_noisy="clean"
    fs=100#TODO: un-hardcode
    n_elec=62
    t_min,t_max=-.1,.4
    lags=np.arange(t_min,t_max+(1/fs),1/fs)
    posarr=utils.get_gtec_pos()
    exclude_list=[] # list of subjects to exclude due to no results file available
    thresh_folds_dir=get_thresh_folds_dir(blocks='6')
    if not grand_avg:
        # do one subject's topomaps
        subj_num="3323"

        subj_cat=utils.get_subj_cat(subj_num)
        # results_fnm=get_subj_results_fnm(results_dir,subj_num)
        
    
        subj_trf_pth=get_subj_trf_pth(subj_num,thresh_folds_dir=thresh_folds_dir,
                                           clean_or_noisy=clean_or_noisy,
                                           rms_str='_rms_',cv_method_str="_nested")
        with open(subj_trf_pth, 'rb') as f:
            trf_results=pickle.load(f)
        weights=trf_results['trf_fitted'].weights.squeeze()
        
        for n_lag, lag_s in enumerate(lags):  # which lag to plot topo for
        # evokeds = trf.to_mne_evoked(montage)[0]
        #TODO: are we going to need the reference electrodes we removed?
            lag_ms=round(1000*lag_s)        
            fig,ax=plt.subplots()
            im,_=plot_topomap(weights[:, n_lag],posarr,axes=ax,show=False,vlim=topo_lims)
            ax.set_title(f'lag={lag_ms} ms')
            fig.colorbar(im,ax=ax)
            plt.show()
            save_dir=os.path.join("..","figures","topos",subj_cat,subj_num)
            save_fnm=f"bkwd_trf_weights_{lag_ms:.0f}_ms"
            save_pth=os.path.join(save_dir, save_fnm)
            if os.path.isdir(save_dir):
                plt.savefig(save_pth)
            else:
                os.makedirs(save_dir,exist_ok=True)
                plt.savefig(save_pth)

    else:
        # get all subjects and plot grand average trf weights
        avg_weights=np.zeros((n_elec,lags.size))
        all_subjs=utils.get_all_subj_nums(single_cat=single_cat)
        #ignore subjects with no results file available
        all_subjs=list(filter(lambda s: s not in exclude_list, all_subjs))
        for subj_num in all_subjs:
            # load each subject's trfs, compute average weights
            subj_cat=utils.get_subj_cat(subj_num)
            # subj_fnm=get_subj_results_fnm(results_dir,subj_num)
            # subj_trf_pth=os.path.join(results_dir,subj_cat,subj_num,subj_fnm)
            subj_trf_pth=get_subj_trf_pth(subj_num,thresh_folds_dir=thresh_folds_dir,
                                           clean_or_noisy=clean_or_noisy,
                                           rms_str='_rms_',cv_method_str="_nested")
            with open(subj_trf_pth, 'rb') as f:
                trf_results=pickle.load(f)
                
            avg_weights+=trf_results['trf_fitted'].weights.squeeze()
            
        avg_weights/=len(all_subjs)
        for n_lag,lag_s in enumerate(lags):  # which lag to plot topo for
        # evokeds = trf.to_mne_evoked(montage)[0]
        #TODO: are we going to need the reference electrodes we removed?
            lag_ms=round(1000*lag_s)
            fig,ax=plt.subplots()
            im,_=plot_topomap(avg_weights[:, n_lag],posarr,axes=ax,show=False,vlim=topo_lims)
            ax.set_title(f'lag={lag_ms} ms')
            fig.colorbar(im,ax=ax)
            plt.show()
            if single_cat is not None:
                save_dir=os.path.join("..","figures","topos",single_cat,"grand_avg")
            else:
                save_dir=os.path.join("..","figures","topos","all_grand_avg")
            save_fnm=f"bkwd_trf_weights_{lag_ms:.0f}ms"
            save_pth=os.path.join(save_dir, save_fnm)
            if os.path.isdir(save_dir):
                plt.savefig(save_pth)
            else:
                os.makedirs(save_dir,exist_ok=True)
                plt.savefig(save_pth)






# %%

#%%
# import packages
import pickle
import scipy.io as spio
import numpy as np
import os
import h5py
from scipy import signal
import utils
from mtrf.model import TRF
import matplotlib.pyplot as plt


# define custom montage position array
from mne.channels import read_custom_montage
from mne.viz import plot_topomap
def get_gtec_pos():
    locs_fl_pth = os.path.join('..',"eeg_data", "raw", 'gtec62.locs')
    montage = read_custom_montage(locs_fl_pth)
    pos = montage.get_positions()
    posarr=np.asarray([xyz[:2] for _y, xyz in pos['ch_pos'].items()])
    return posarr
#%%
# skeleton code for topoplot

if __name__=='__main__':
    grand_avg=True
    fs=100#TODO: un-hardcode
    n_elec=62
    n_lags=41
    posarr=get_gtec_pos()
    #TODO: how to add a colorbar? and how to normalize it?
    if not grand_avg:
        # do one subject's topomaps
        subj_num="3253"
        subj_cat=utils.get_subj_cat(subj_num)
        subj_trf_pth=os.path.join("..", "results", subj_cat, subj_num,
                                subj_num+"_clean_pauses_env_recon_results.pkl")
        with open(subj_trf_pth, 'rb') as f:
            trf_results=pickle.load(f)
        weights=trf_results['trf_fitted'].weights.squeeze()
        
        for n_lag in range(weights.shape[1]):  # which lag to plot topo for
        # evokeds = trf.to_mne_evoked(montage)[0]
        #TODO: are we going to need the reference electrodes we removed?
            fig,ax=plt.subplots()
            plot_topomap(weights[:, n_lag], posarr,axes=ax,show=False)
            ax.set_title(f'lag={1000*n_lag/fs} ms')
            plt.show()
    else:
        # get all subjects and plot grand average trf weights
        avg_weights=np.zeros((n_elec,n_lags))
        hc_pth=os.path.join("..", "results","hc")
        sp_pth=os.path.join("..", "results","sp")
        all_subjs=os.listdir(hc_pth)+os.listdir(sp_pth)
        for subj_num in all_subjs:
    
            subj_cat=utils.get_subj_cat(subj_num)
            subj_trf_pth=os.path.join("..", "results", subj_cat, subj_num,
                                  subj_num+"_clean_pauses_env_recon_results.pkl")
            with open(subj_trf_pth, 'rb') as f:
                trf_results=pickle.load(f)
            avg_weights+=trf_results['trf_fitted'].weights.squeeze()
            
        avg_weights/=len(all_subjs)
        for n_lag in range(avg_weights.shape[1]):  # which lag to plot topo for
        # evokeds = trf.to_mne_evoked(montage)[0]
        #TODO: are we going to need the reference electrodes we removed?
            fig,ax=plt.subplots()
            plot_topomap(avg_weights[:, n_lag], posarr,axes=ax,show=False)
            ax.set_title(f'lag={1000*n_lag/fs} ms')
            plt.show()





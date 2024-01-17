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
#%%
#%%
# define custom montage position array
from mne.channels import read_custom_montage
from mne.viz import plot_topomap
def get_gtec_pos():
    locs_fl_pth = os.path.join('..',"eeg_data", "raw", 'gtec62.locs')
    montage = read_custom_montage(locs_fl_pth)
    pos = montage.get_positions()
    posarr=np.asarray([xyz[:2] for key, xyz in pos['ch_pos'].items()])
    return posarr
#%%
# skeleton code for topoplot
# TODO: remove this stuff
if __name__=='__main__':

    
    subj_num="3253"
    subj_cat=utils.get_subj_cat(subj_num)
    subj_trf_pth=os.path.join("..", "results", subj_cat, subj_num,
                              subj_num+"_clean_pauses_env_recon_results.pkl")
    with open(subj_trf_pth, 'rb') as f:
        trf_results=pickle.load(f)
    weights=trf_results['trf_fitted'].weights
    posarr=get_gtec_pos()
    for n_lag in range(weights.shape[1]):  # which lag to plot topo for
    # evokeds = trf.to_mne_evoked(montage)[0]
    #TODO: are we going to need the reference electrodes we removed?
        plot_topomap(weights.squeeze()[:, n_lag], posarr)



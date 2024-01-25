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
# topoplotting script

if __name__=='__main__':
    # set colorbar upper and lower bounds
    topo_lims=(-0.028, 0.028)

    grand_avg=True
    #only matters when grand_avg is True, select only sp or 
    # only hc, else None to do all

    single_cat="hc"
    fs=100#TODO: un-hardcode
    n_elec=62
    n_lags=41
    posarr=get_gtec_pos()
    if not grand_avg:
        # do one subject's topomaps
        subj_num="3328"

        subj_cat=utils.get_subj_cat(subj_num)
        subj_trf_pth=os.path.join("..","results",subj_cat,subj_num,
                                subj_num+"_clean_pauses_env_recon_results.pkl")
        with open(subj_trf_pth, 'rb') as f:
            trf_results=pickle.load(f)
        weights=trf_results['trf_fitted'].weights.squeeze()
        
        for n_lag in range(weights.shape[1]):  # which lag to plot topo for
        # evokeds = trf.to_mne_evoked(montage)[0]
        #TODO: are we going to need the reference electrodes we removed?
            lag_ms=1000*n_lag/fs
            fig,ax=plt.subplots()
            im,_=plot_topomap(weights[:, n_lag],posarr,axes=ax,show=False,vlim=topo_lims)
            ax.set_title(f'lag={1000*n_lag/fs} ms')
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
        avg_weights=np.zeros((n_elec,n_lags))
        if single_cat is None:
            hc_pth=os.path.join("..", "results","hc")
            sp_pth=os.path.join("..", "results","sp")
            all_subjs=os.listdir(hc_pth)+os.listdir(sp_pth)
        elif single_cat=='hc':
            hc_pth=os.path.join("..", "results","hc")
            all_subjs=os.listdir(hc_pth)
        elif single_cat=='sp':
            sp_pth=os.path.join("..", "results","sp")
            all_subjs=os.listdir(sp_pth)
        else:
            raise NotImplementedError(f'{single_cat} is not an option for <single_cat>')
        for subj_num in all_subjs:
            # load each subject's trfs, compute average weights
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
            lag_ms=1000*n_lag/fs
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






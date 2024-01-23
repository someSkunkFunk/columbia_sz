#%%
# import packages
import pickle
import numpy as np
import os

import utils
from mtrf.model import TRF
import matplotlib.pyplot as plt

#%%
# boxplotting script

if __name__=='__main__':
    # set colorbar upper and lower bounds
    
    fs=100#TODO: un-hardcode
    n_elec=62
    n_lags=41
    # get all subjects and plot grand average trf weights
    avg_weights=np.zeros((n_elec,n_lags))
    hc_pth=os.path.join("..", "results","hc")
    sp_pth=os.path.join("..", "results","sp")
    all_subjs=os.listdir(hc_pth)+os.listdir(sp_pth)
    # initialize r-values vectors
    hc_rs=np.zeros(len(os.listdir(hc_pth)))
    sp_rs=np.zeros(len(os.listdir(sp_pth)))


    for subj_num in all_subjs:
        # load each subject's trfs, compute average weights
        subj_cat=utils.get_subj_cat(subj_num)
        subj_trf_pth=os.path.join("..", "results", subj_cat, subj_num,
                                subj_num+"_clean_pauses_env_recon_results.pkl")
        with open(subj_trf_pth, 'rb') as f:
            trf_results=pickle.load(f)
        
        if subj_cat=='hc':
            hc_rs[np.where(hc_rs==0)[0][0]]=trf_results['r_ncv'].mean()
        elif subj_cat=="sp":
            sp_rs[np.where(sp_rs==0)[0][0]]=trf_results['r_ncv'].mean()

    fig,ax=plt.subplots()
    ax.boxplot((hc_rs,sp_rs), labels=("hc", "sp"))
    ax.set_title('5fold-cv mean recontruction accuracies')
    ax.set_ylabel('mean r')
    plt.show()
    save_dir=os.path.join("..","figures","boxplots","all")
    save_fnm=f"bkwd_trf_mean_ncvrs"
    save_pth=os.path.join(save_dir, save_fnm)
    if os.path.isdir(save_dir):
        plt.savefig(save_pth)
    else:
        os.makedirs(save_dir,exist_ok=True)
        plt.savefig(save_pth)






# %%

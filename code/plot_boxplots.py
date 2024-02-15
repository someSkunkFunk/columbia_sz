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
    results_dir=os.path.join("..","results")
    # get all subjects and plot grand average trf weights
    avg_weights=np.zeros((n_elec,n_lags))
    hc_subjs=utils.get_all_subj_nums(single_cat="hc")
    sp_subjs=utils.get_all_subj_nums(single_cat="sp")
    all_subjs=utils.get_all_subj_nums() 
    # initialize r-values vectors
    hc_rs=np.zeros(len(hc_subjs))
    sp_rs=np.zeros(len(sp_subjs))


    for subj_num in all_subjs:
        # load each subject's trfs, compute average weights
        subj_cat=utils.get_subj_cat(subj_num)
        
        results_fnm='bkwd_trf.pkl'
        
        subj_trf_pth=os.path.join(results_dir,subj_cat,subj_num,results_fnm)
        with open(subj_trf_pth, 'rb') as f:
            trf_results=pickle.load(f)
        # set first zero-valued element in arrya to mean of current subject, depending on category
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

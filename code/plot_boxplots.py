#%%
# INIT
import pickle
import numpy as np
import os

import utils
from mtrf.model import TRF
import matplotlib.pyplot as plt

def get_subj_trf_pth(subj_num,thresh_folds_dir,clean_or_noisy,rms_str,cv_method_str,use_decimate):
    subj_cat=utils.get_subj_cat(subj_num)
        
    results_fnm=f'bkwd_trf_{clean_or_noisy}{rms_str}stims{cv_method_str}.pkl'
    if use_decimate:
        evnt_dir="evnt_decimate"
    else:
        evnt_dir="evnt"
    subj_results_dir=os.path.join("..","results",evnt_dir,
                                thresh_folds_dir,subj_cat,subj_num)
    subj_trf_pth=os.path.join(subj_results_dir,results_fnm)
    return subj_trf_pth
#%%
# MAIN BOXPLOT SCRIPT

if __name__=='__main__':
    #specify results to load
    use_decimate=True
    evnt=True
    shuffled_trials=True
    rm_old_figs=True
    blocks='1,2,3,4,5,6' # all or list of numbers as a string
    evnt_thresh='750'
    k=5 #number of cv folds
    clean_or_noisy='noisy'
    rms_str='_rms_' # '_rms_' or ''
    cv_method_str='_nested' #"_nested" or "_crossval"
    


    ylims=[-0.02, 0.08] # vertical axis limits
    if shuffled_trials:
        shuffled="shuffled"
    else:
        shuffled="shuffless"

    if evnt:
        timestamps_generated_by="evnt"
        thresh_dir=f"thresh_{evnt_thresh}"
        if k!=-1:
            thresh_folds_dir=thresh_dir+f"_{k}fold"+f"_{shuffled}"
        elif k==-1:
            thresh_folds_dir=thresh_dir+"_loo"+f"_{shuffled}"
        if blocks!="all" and blocks!="1,2,3,4,5,6":
            blocks_to_keep=['b0'+b.strip() for b in blocks.split(",")]
            print("note: need to change this in xcorr case")
            blocks_str="".join(blocks_to_keep)
            thresh_folds_dir=thresh_folds_dir+"_"+blocks_str
            blocks_title_str=f" - {blocks_str} only"
        else:
            # for plot title
            blocks_title_str=" - all blocks"


        
        

    if evnt:
        # results_dir=os.path.join("..","results","evnt",thresh_dir)
        if use_decimate:
            decimate_str="_decimate"
        else:
            decimate_str=""
        save_dir=os.path.join("..","figures","boxplots",thresh_folds_dir+decimate_str)
        if rm_old_figs:
            utils.rm_old_figs(save_dir)
    else:
        raise NotImplementedError('gotta fix this')
        # results_dir=os.path.join("..","results",???)
        # save_dir=os.path.join("..","figures","boxplots","all")


    fs=100#TODO: un-hardcode
    n_elec=62
    n_lags=41
    
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
        subj_trf_pth=get_subj_trf_pth(subj_num,thresh_folds_dir,clean_or_noisy,rms_str,cv_method_str,use_decimate)
        with open(subj_trf_pth, 'rb') as f:
            trf_results=pickle.load(f)
        # set first zero-valued element in arrya to mean of current subject, depending on category
        if subj_cat=='hc':
            hc_rs[np.where(hc_rs==0)[0][0]]=trf_results['r_ncv'].mean()
        elif subj_cat=="sp":
            sp_rs[np.where(sp_rs==0)[0][0]]=trf_results['r_ncv'].mean()

    fig,ax=plt.subplots()
    ax.boxplot((hc_rs,sp_rs), labels=("hc", "sp"))
   
    if shuffled_trials:
        shuff_str="shuffled"
    else:
        shuff_str="not shuffled"
    ax.set_title(f'{k}fold mean recontruction accuracies using {clean_or_noisy} stims ({shuff_str})'+blocks_title_str)
   
    ax.set_ylabel('mean r')
    ax.set_ylim(ylims[0], ylims[1])
    
    save_fnm=f"bkwd_trf_recons_{clean_or_noisy}_stims"
    save_pth=os.path.join(save_dir,save_fnm)
    if os.path.isdir(save_dir):
        plt.savefig(save_pth)
    else:
        os.makedirs(save_dir,exist_ok=True)
        plt.savefig(save_pth)

    plt.show()




# %%

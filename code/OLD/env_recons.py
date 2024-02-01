# MAIN ANALYSIS script for backward mtrf model 
# on columbia sz dataset 
# using as blueprint for bluehive job script
# most up-to-date as of 9/26/23
#%%
# import packages
import pickle
import scipy.io as spio
import numpy as np
import os

import utils
from trf_helpers import find_bad_electrodes, get_stim_envs, setup_xy
from mtrf.model import TRF
from mtrf.stats import crossval, nested_crossval


#%%
# define functions


def nested_cv_wrapper(subj_num,
                      direction=-1,
                      tmin=0,
                      tmax=0.4,
                      k=5,
                      lim_stim=None,
                      save_results=False,
                      drop_bad_electrodes=False,
                      clean_nxor_noisy=['clean'], 
                      regs=np.logspace(-1, 8, 10),
                      reduce_trials_by = "pauses",
                      return_xy=False,
                      **kwargs):
    '''
    direction: 1 -> encoder -1 -> decoder
    lim_stim: limit to n number of stimuli for faster execution (won't save result)
    clean_nxor_noisy: non-exclusive or choose cleanr or noisy (list, turned into list if not entered as list but elements must be strings)
    drop_bad_electrodes: drop electrodes that are "outliers"
    regs: ridge regularization params to optimize over
    reduce_trials_by: str specifying  by grouping stories wihtin a block ("stim_nm")
    # or grouping by pauses within a block ("pauses")
    '''

    subj_cat = utils.get_subj_cat(subj_num)
    subj_data = utils.load_preprocessed(subj_num)
    stims_dict = utils.load_stims_dict()
    fs_trf = subj_data['fs'][0] # use eeg fs, assumed eeg downsampled to desired fs
    if lim_stim is not None:
        # in case we want to run to completion for testing 
        print(f'running number of stimuli limited to {lim_stim}, wont save result...\n')
    if (lim_stim is not None) and save_results:
        raise NotImplementedError("not saving results by default because assuming we just want to test that this runs.") 

    if isinstance(clean_nxor_noisy, str):
        # if single string given instead of list
        clean_nxor_noisy = [clean_nxor_noisy]
          # 
    if drop_bad_electrodes:
        outlier_idx = find_bad_electrodes(subj_data)
    else:
        outlier_idx = None

    for clean_or_noisy in clean_nxor_noisy:
        stim_envs = get_stim_envs(stims_dict, clean_or_noisy, fs_output=fs_trf)
        stimulus, response, stim_nms = setup_xy(subj_data, stim_envs, subj_cat, subj_num, reduce_trials_by, outlier_idx)
        # model params
        trf = TRF(direction=direction)  # use bkwd model
        


        r_ncv, best_lam = nested_crossval(trf, stimulus[:lim_stim], response[:lim_stim], fs_trf, tmin, tmax, regs, k=k)
        


        if lim_stim is None and save_results:
            # save results

            results_file = "_".join([subj_num, clean_or_noisy, reduce_trials_by])+"_env_recon_results.pkl"
            results_dir = os.path.join("..","results", subj_cat, subj_num)
            # Check if the directory exists; if not, create it
            # note: will also create parent directories
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, results_file), 'wb') as f:
                pickle.dump({'trf_fitted': trf, 'r_ncv': r_ncv, 'best_lam': best_lam,
                                'stimulus': stimulus, 'response': response, 'stim_nms': stim_nms}, f)


    if return_xy == True:
        return trf, r_ncv, best_lam, (stimulus, response, stim_nms)
    else:
        return trf, r_ncv, best_lam
#%%
# test script
if __name__=='__main__':
    subj_num = '0194'
    # trf, r, lam = nested_cv_wrapper(subj_num, lim_stim=6)
    trf, r, lam, (S, R, nms) = nested_cv_wrapper(subj_num, lim_stim=6, return_xy=True, save_results=False)   



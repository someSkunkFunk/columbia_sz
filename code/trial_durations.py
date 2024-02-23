# script for looking at trial durations 
#%%
# import packages
import pickle
import scipy.io as spio
import numpy as np
import os
import matplotlib.pyplot as plt

import utils
from trf_helpers import get_stim_envs, setup_xy
#%%
# set up
all_subjs=utils.get_all_subj_nums()
which_xcorr="wavs" # clean wavs seem to yield best results....
stims_dict=utils.get_stims_dict()
all_stim_nms=stims_dict["ID"] #note .wav not in name
f_lp=49 #Hz, lowpass filter freq for get_stim_envs
evnt=False
reduce_trials_by='pauses'
clean_or_noisy='clean'
#%%
blocks=[f"B0{ii}" for ii in range(1,7)]
# store trial durations in dict organized by blocks
trial_durations_dict={blk: [] for blk in blocks}
for subj_num in all_subjs:
    print(f"aggregating subj {subj_num} data.")
    subj_cat=utils.get_subj_cat(subj_num)
    #NOTE that  by leaving eeg_dir blank below it's looking 
    # in eeg_data/preproessed_xcorr by default
    subj_data=utils.load_preprocessed(subj_num,which_xcorr=which_xcorr)
    fs_trf=subj_data['fs'][0]
    # note: getting stim envs kinda unnecessary and time consuming for this analysis
    # so maybe can edit setup_xy so it doesn't need it?
    stim_envs=get_stim_envs(stims_dict,clean_or_noisy,fs_output=fs_trf,f_lp=f_lp)
    stimulus,_,stim_nms,_=setup_xy(subj_data,stim_envs,subj_num,
                                                       reduce_trials_by,which_xcorr=which_xcorr)
    for stim_ii, nms in enumerate(stim_nms):
        # note that nms is a list of names, but all nms for a given block
        # should belong to the same trial
        block=nms[0][:3].capitalize()
        trial_dur=(stimulus[stim_ii].size-1)/fs_trf
        trial_durations_dict[block].append(trial_dur)
#%%
#plotting
# make a summary bar graph

# make histograms for each block
fig, axs=plt.subplots(6,1,tight_layout=True)
n_bins=40
for ii, block in enumerate(blocks):
    median=np.median(trial_durations_dict[block])
    axs[ii].hist(trial_durations_dict[block],bins=n_bins)
    axs[ii].set_title(f'{block}')
    axs[ii].axvline(median,label=f'median: {median:.3f} s', color='red')
    axs[ii].set_xlabel('time,s')
    axs[ii].legend()
plt.show()
for ii, block in enumerate(blocks):
    print(f"""min/max duration in block {ii+1}: {min(trial_durations_dict[block]),
                                               max(trial_durations_dict[block])}s""")
    # print(f"max duration in block {ii+1}: {max(trial_durations_dict[block])}s")
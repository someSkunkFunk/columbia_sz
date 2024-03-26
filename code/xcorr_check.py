# script to check that stim envelopes actually give anything when using xcorr
#%%
# import packages
import pickle
import scipy.io as spio
import numpy as np
import os
import matplotlib.pyplot as plt

import utils
from trf_helpers import load_stim_envs, setup_xy
from scipy import signal 
#%%
# set up
all_subjs=utils.get_all_subj_nums()
stims_dict=utils.get_stims_dict()
all_stim_nms=stims_dict["ID"] #note .wav not in name
f_lp=49 #Hz, lowpass filter freq for get_stim_envs
evnt=True
if evnt:
    which_xcorr=None 
    reduce_trials_by=None
    thresh_dir='thresh_750'
    eeg_dir=os.path.join("..","eeg_data","preprocessed_evnt",thresh_dir)
else:
    eeg_dir=None
    which_xcorr='wavs'
    reduce_trials_by='pauses'
clean_or_noisy='clean'
#%%
# plot xcorrs for each subj, stimulus, electrode
blocks=[f"B0{ii}" for ii in range(1,7)]
# store trial durations in dict organized by blocks
corrs_dict={sbj: () for sbj in all_subjs}
fs_trf=100
stim_envs=load_stim_envs()
for subj_num in all_subjs:
    print(f"aggregating subj {subj_num} data....")
    subj_cat=utils.get_subj_cat(subj_num)
    #NOTE that  by leaving eeg_dir blank below it's looking 
    # in eeg_data/preproessed_xcorr by default
    subj_data=utils.load_preprocessed(subj_num,
                                      eeg_dir=eeg_dir,
                                      which_xcorr=which_xcorr,
                                      evnt=evnt)
    # fs_trf=subj_data['fs'][0]
    # note: getting stim envs kinda unnecessary and time consuming for this analysis
    # so maybe can edit setup_xy so it doesn't need it?
    stimulus,response,stim_nms,audio_recorded=setup_xy(subj_data,stim_envs,subj_num,evnt=evnt,
                                                       reduce_trials_by=reduce_trials_by,
                                                       which_xcorr=which_xcorr)
    
    # pad stim and response to same length for aveeraging purposes
    max_len=max([len(s) for s in stimulus])
    padded_stimulus=[np.pad(s, (0,max_len-len(s))) for s in stimulus]
    padded_response=[np.pad(rs, (0,max_len-len(rs.squeeze()))) for rs in response]
    subj_xcorrs=np.zeros((max_len,len(stimulus),response[0].shape[1]))#time x stims x electrodes
    # -> time x electrodes after averaging over stims
    for ii, (s,rs) in enumerate(zip(padded_stimulus,padded_response)):
        # subj_xcorrs.append([])
        print(f'xcorring stim {ii+1} of {len(stimulus)} ...')
        # lags should be the same for all electrodes
        lags=signal.correlation_lags(s.size,len(rs),mode='full')
        for jj, r in enumerate(rs.T):
            subj_xcorrs[:,ii,jj]=signal.correlate(r,s,mode='full')
    # average out stimuli
    subj_xcorrs=subj_xcorrs.mean(axis=1).squeeze()
    raise NotImplementedError('stop here.')
    # plot the shit
    fig,ax=plt.subplots()
    ax.plot(lags/fs_trf,r_xcorr)
    #TODO: get electrode name from gtec.pos
    ax.set_title(f'electrode {elec_num} xcorr for stim/subj {elec_num, subj_num}')
    ax.set_xlabel('lag (s)')
    # ax.set_xlim([-.200,.400]) #TODO: where to expect trf??
    save_dir=os.path.join("..","figures",
                            "evnt_info",thresh_dir,"xcorr_align_check",
                            subj_cat,subj_num)
    if not evnt:
        raise NotImplementedError("need to update output figures folder structure for xcorr-derived timestamp results")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    fig_nm=f"xcorr_chn_{jj:0{2}}"
    save_pth=os.path.join(save_dir,fig_nm)
    plt.savefig(save_pth)
    # plt.show()
del lags,r_xcorr,s,r,fig,ax
    del subj_data,stimulus,response

            
    # corrs_dict[subj_num]=(lags,subj_xcorrs) # {subj_num: [[stims x channels]]}
#%%
# plotting 
subj_num="3253"
subj_cat=utils.get_subj_cat(subj_num)
stim_ii=0 # choose some stimm to plot
lags,subj_xcorrs=corrs_dict[subj_num]
for chnl in range(62):
    

    

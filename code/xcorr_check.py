# script to check that stim envelopes actually give anything when using xcorr
#%%
# INIT
import pickle
import scipy.io as spio
import numpy as np
import os
import matplotlib.pyplot as plt

import utils
from trf_helpers import load_stim_envs, setup_xy
from scipy import signal 

# set up
all_subjs=utils.get_all_subj_nums()
stims_dict=utils.get_stims_dict()
all_stim_nms=stims_dict["ID"] #note .wav not in name
# f_lp=49 #Hz, lowpass filter freq for get_stim_envs
evnt=True
use_rms=True
if evnt:
    which_xcorr=None 
    reduce_trials_by=None
    thresh_dir='thresh_750'
    eeg_dir=os.path.join("..","eeg_data","preprocessed_decimate",thresh_dir)
else:
    eeg_dir=None
    which_xcorr='wavs'
    reduce_trials_by='pauses'
clean_or_noisy='noisy'
#%%
# plot xcorrs for each subj, stimulus, electrode
blocks=[f"B0{ii}" for ii in range(1,7)]
# store trial durations in dict organized by blocks
#NOTE: forgot why we wanted this but maybe still useful?
# corrs_dict={sbj: () for sbj in all_subjs}
fs_trf=100
if use_rms:
    # filtered at 1-15 Hz
    stim_envs=utils.load_matlab_envs(clean_or_noisy)
else:
    stim_envs=load_stim_envs(clean_or_noisy)
# list of values to use as plt.set_xlim for zoomed views on figs
zoom_lvls=[[-0.2,.5],[-.400,.200], [-.500,.500], [-1.0,0.6], [-5,5]]

for subj_num in all_subjs:
    print(f"aggregating subj {subj_num} data....")
    subj_cat=utils.get_subj_cat(subj_num)
    #NOTE that  by leaving eeg_dir blank below it's looking 
    # in eeg_data/preproessed_xcorr by default
    subj_data=utils.load_preprocessed(subj_num,
                                      eeg_dir=eeg_dir,
                                      which_xcorr=which_xcorr,
                                      evnt=evnt)
    stimulus,response,stim_nms,_=setup_xy(subj_data,stim_envs,subj_num,evnt=evnt,
                                                       reduce_trials_by=reduce_trials_by,
                                                       which_xcorr=which_xcorr)
    
    # pad stim and response to same length for aveeraging purposes
    max_len=max([len(s) for s in stimulus])
    padded_stimulus=[np.pad(s, (0,max_len-len(s))) for s in stimulus]
    padded_response=[np.pad(rs, ((0,max_len-len(rs.squeeze())), (0,0)) ) for rs in response]
    subj_xcorrs=np.zeros((max_len*2-1,len(stimulus),response[0].shape[1]))#time x stims x electrodes
    # -> time x electrodes after averaging over stims
    for ii, (s,rs) in enumerate(zip(padded_stimulus,padded_response)):
        # subj_xcorrs.append([])
        print(f'xcorring stim {ii+1} of {len(stimulus)} ...')
        # lags should be the same for all electrodes AND stimuli (since padded)
        if ii==0:
            lags=signal.correlation_lags(s.size,len(rs),mode='full')
        for jj, r in enumerate(rs.T):
            subj_xcorrs[:,ii,jj]=signal.correlate(r,s,mode='full')
    # # average out stimuli
    # subj_xcorrs=subj_xcorrs.mean(axis=1).squeeze()
    figs_dir=os.path.join("..","figures",
                            "evnt_info",thresh_dir,"xcorr_align_check_decimate",
                            subj_cat,subj_num)
    utils.rm_old_figs(figs_dir)#NOTE: CAREFUL NOT TO SPECIFY PARENT DIRECTORY CAUSING UNRELATED FIGURES
    # TO BE DELETED SINCE THIS FUNCTION REMOVES FILES IN SUB-DIRECTORIES ALSO
    for electrode_index in range(subj_xcorrs.shape[-1]):
        print(f"plottin electrode: {electrode_index+1} of 62...")
        for stim_index in range(subj_xcorrs.shape[1]):
            print(f"stim {stim_index+1} of {subj_xcorrs.shape[1]}...")
            segment_nms=stim_nms[stim_index]
            xcorr_vals=subj_xcorrs[:,stim_index,electrode_index]
            # plot the shit
            fig,ax=plt.subplots()
            ax.plot(lags/fs_trf,xcorr_vals)
            #TODO: get electrode name from gtec.pos
            ax.set_title(f'(electrode,subj,stims): {electrode_index, subj_num,segment_nms}')
            ax.set_xlabel('lag (s)')
            
            if not evnt:
                raise NotImplementedError("need to update output figures folder structure for xcorr-derived timestamp results")
            if not os.path.isdir(figs_dir):
                os.makedirs(figs_dir,exist_ok=True)
            fig_nm=f"xcorr_chn_{electrode_index:0{2}}_{stim_index:0{3}}_{clean_or_noisy}"
            save_pth=os.path.join(figs_dir,fig_nm)
            plt.savefig(save_pth)
            # zoom in and save zoomed version for readability
            for ii,lvl in enumerate(zoom_lvls):
                ax.set_xlim(lvl) #TODO: where to expect trf??
                plt.savefig(save_pth+f"_zoomed_{ii}")
            plt.close()
        # plt.show()
    #     del avg_xcorr,s,r,fig,ax
    # del subj_data,stimulus,response,lags,padded_stimulus,padded_response

            

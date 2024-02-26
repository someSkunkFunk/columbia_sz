# script to check that stim envelopes actually give anything when using xcorr
#%%
# import packages
import pickle
import scipy.io as spio
import numpy as np
import os
import matplotlib.pyplot as plt

import utils
from trf_helpers import get_stim_envs, setup_xy
from scipy import signal 
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
# plot xcorrs for each subj, stimulus, electrode
blocks=[f"B0{ii}" for ii in range(1,7)]
# store trial durations in dict organized by blocks
corrs_dict={sbj: () for sbj in all_subjs}
fs_trf=100
stim_envs=get_stim_envs(stims_dict,clean_or_noisy,fs_output=fs_trf,f_lp=f_lp)
    
for subj_num in all_subjs:
    print(f"aggregating subj {subj_num} data....")
    subj_cat=utils.get_subj_cat(subj_num)
    #NOTE that  by leaving eeg_dir blank below it's looking 
    # in eeg_data/preproessed_xcorr by default
    subj_data=utils.load_preprocessed(subj_num,which_xcorr=which_xcorr)
    # fs_trf=subj_data['fs'][0]
    # note: getting stim envs kinda unnecessary and time consuming for this analysis
    # so maybe can edit setup_xy so it doesn't need it?
    stimulus,response,_,_=setup_xy(subj_data,stim_envs,subj_num,
                                                       reduce_trials_by,which_xcorr=which_xcorr)
    # subj_xcorrs=[]
    for ii, (s,rs) in enumerate(zip(stimulus,response)):
        # subj_xcorrs.append([])
        print(f'xcorring stim {ii+1} of {len(stimulus)} ...')
        # lags should be the same for all electrodes
        lags=signal.correlation_lags(s.size,len(rs),mode='full')
        for jj, r in enumerate(rs.T):
            # subj_xcorrs[ii].append(signal.correlate(s,r,mode='valid'))
            r_xcorr=signal.correlate(r,s,mode='full')
            # plot the shit
            fig,ax=plt.subplots()
            ax.plot(lags/fs_trf,r_xcorr)
            ax.set_title(f'electrode {jj} xcorr for stim/subj {ii, subj_num}')
            ax.set_xlabel('time (s)')
            # ax.set_xlim([-.200,.400]) #TODO: where to expect trf??
            save_dir=os.path.join("..","figures","xcorrs",subj_cat,subj_num)
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
    

    

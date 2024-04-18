# script to implement various forms of correlation amongst the aligned eeg data to see if 
# there is really any misalignment OR something about how we conduct the TRF analysis is 
# giving poor envelope reconstructions
# assumes we just looking at evnt


#NOTE: realizing may be easier to deal with variable trial lengths by re-preprocessing the eeg 
# and comparing at the individual wav file level
#%%
import os
import utils
import numpy as np

#%%
# STEP ONE: LOAD ALL SUBJECT DATA & EVNT info
def load_all_subj_data(evnt_thresh='000'):
    _limit_subjs=True
    thresh_dir=f"thresh_{evnt_thresh}" # should be three digit number representing decimals to third place
    prep_data_dir=os.path.join("..","eeg_data","preprocessed_evnt",thresh_dir)
    if _limit_subjs:
        amt_subjs=1
        print("only loading a single subject. set _limit_subjs to False to load all subjects")     
    else:
        amt_subjs=20 #number of subjs
    subj_ints=[ii for ii in range(amt_subjs)]
    all_subj_data={}
    for subj_int in subj_ints:
        subj_num=utils.assign_subj(subj_int)
        all_subj_data[subj_num]=utils.load_preprocessed(subj_num,eeg_dir=prep_data_dir,
                                                evnt=True,which_xcorr=None)
    return all_subj_data

#%%
# STEP TWO: use evnt info to slice preprocessed eeg 
# by matching names

#%%
# STEP 3: GROUP TIMESTAMPS INTO TRIALS BASED ON PAUSES
# then recalculate onsets/offsets relative to start of trial?
def split_eeg_trials_to_wavs(all_subj_data,evnt_nms,evnt_onsets,evnt_offsets):
    '''
    splits trials of eeg back into eeg for each individual wav file
    assumes evnt info specified in same fs as all_subj_data
    doesn't track what subject eeg belongs to, but I don't think it matters here...
    can easily add that if needed though
    returns
    eeg_by_wavs=dict('stim_wav_nm': [eeg for each subject that has response])
    '''
    eeg_by_wavs={}
    for subj, subj_data in all_subj_data.items():
        print(f'splitting subj {subj} trials into individual stims')
        for trial_num, (_, data) in enumerate(subj_data.items()):
            print(f"starting trial {trial_num+1} of {len(subj_data)}.")
            nms_in_trial=data[0]
            eeg=data[2]
            trial_start=evnt_onsets[evnt_nms==nms_in_trial[0]]
            for nm in nms_in_trial:
                # get onset/offset relative to trial start
                stim_trial_onset=(evnt_onsets[evnt_nms==nm]-trial_start)[0]
                stim_trial_offset=(evnt_offsets[evnt_nms==nm]-trial_start)[0]
                # note: indexing to get int instead of array 
                if nm in eeg_by_wavs:
                    eeg_by_wavs[nm].append(eeg[stim_trial_onset:stim_trial_offset])
                else:
                    eeg_by_wavs[nm]=[eeg[stim_trial_onset:stim_trial_offset]]
            

    return eeg_by_wavs

# STEP 4: CALCULATE SPLIT-HALF CORRELATION
from scipy.stats import pearsonr
def split_half_corr(eeg_by_wavs):
    corrs_by_wavs={}
    pvals_by_wavs={}
    for stim_nm, all_subjs_eeg in eeg_by_wavs.items():
        n_subjs=len(all_subjs_eeg)
        corr,pval=pearsonr(all_subjs_eeg[:n_subjs//2],all_subjs_eeg[n_subjs//2:])
        corrs_by_wavs[stim_nm]=corr
        pvals_by_wavs[stim_nm]=pval



    return corrs_by_wavs,pvals_by_wavs


#%% full script
if __name__=='__main__':

    all_subj_data=load_all_subj_data()
    fs_trf=100 # evnt times in seconds; use trf sampling rate for preprocessed data
    sample_subj=[k for k in all_subj_data.keys()][0]
    print(f"loading timestamps for subject {sample_subj}")
    evnt=utils.load_evnt(sample_subj)
    evnt_nms,evnt_blocks,evnt_confidence,evnt_onsets,evnt_offsets=utils.extract_evnt_data(evnt,fs_trf)
    eeg_by_wavs=split_eeg_trials_to_wavs(all_subj_data,evnt_nms,evnt_onsets,evnt_offsets)
    corrs_by_wavs,pvals_by_wavs=split_half_corr(eeg_by_wavs)
    



#OLD IDEA:
    
# STEP TWO: USE SEGMENT NAMES TO PARTITION DATA INTO TRIALS USING SET OPERATIONS
# MIGHT BE WORTHWHILE TO CHECK OUR ASSUMPTION ABOUT ALL TRIALS BELONGING TO THE SAME SET
# first, isolate names of wav files in each trial from all subjects 
    

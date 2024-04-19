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
# DEFINE LOCAL FUNCTIONS
# STEP ONE: LOAD ALL SUBJECT DATA & EVNT info
def load_all_subj_data(evnt_thresh='000'):
    _limit_subjs=False
    thresh_dir=f"thresh_{evnt_thresh}" # should be three digit number representing decimals to third place
    prep_data_dir=os.path.join("..","eeg_data","preprocessed_evnt",thresh_dir)
    if _limit_subjs:
        _amt_subjs=2
        print(f"only loading a {_amt_subjs} subject(s). set _limit_subjs to False to load all subjects")     
    else:
        print(f"loading all subject data.")
        _amt_subjs=20 #number of subjs
    subj_ints=[ii for ii in range(_amt_subjs)]
    all_subj_data={}
    for subj_int in subj_ints:
        subj_num=utils.assign_subj(subj_int)
        all_subj_data[subj_num]=utils.load_preprocessed(subj_num,eeg_dir=prep_data_dir,
                                                evnt=True,which_xcorr=None)
    return all_subj_data
def trim_lengths_to_match(all_subjs_eeg_same_stim):
    min_length=min([resp.shape[0] for resp in all_subjs_eeg_same_stim])
    trimmed_eeg=[resp[:min_length,:] for resp in all_subjs_eeg_same_stim]
    return trimmed_eeg
# STEP TWO: use evnt info to slice preprocessed eeg 
# by matching names


# STEP 3: GROUP TIMESTAMPS INTO TRIALS BASED ON PAUSES
# then recalculate onsets/offsets relative to start of trial?
def split_eeg_trials_to_wavs(all_subj_data,fs):
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
        print(f"loading timestamps for subject {subj}")
        evnt=utils.load_evnt(subj)
        evnt_nms,_,_,evnt_onsets,evnt_offsets=utils.extract_evnt_data(evnt,fs_trf)
        for trial_num, (_, data) in enumerate(subj_data.items()):
            print(f"starting trial {trial_num+1} of {len(subj_data)}.")
            nms_in_trial=data[0]
            eeg=data[2]
            trial_start=evnt_onsets[evnt_nms==nms_in_trial[0]]
            for nm in nms_in_trial:
                # get onset/offset relative to trial start
                try:
                    stim_trial_onset=(evnt_onsets[evnt_nms==nm]-trial_start)[0]
                    stim_trial_offset=(evnt_offsets[evnt_nms==nm]-trial_start)[0]
                except:
                    pass
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
    n_electrodes=62
    for stim_nm, all_subjs_eeg in eeg_by_wavs.items():
        
        n_subjs=len(all_subjs_eeg)
        if n_subjs<=1:
            # drop stim since no correlation to compute
            continue
        all_subjs_trimmed_eeg=trim_lengths_to_match(all_subjs_eeg)
        first_split=np.asarray(all_subjs_trimmed_eeg[:n_subjs//2])
        first_split_mean=first_split.mean(axis=0)
        second_split=np.asarray(all_subjs_trimmed_eeg[n_subjs//2:])
        second_split_mean=second_split.mean(axis=0)
        corrs_by_wavs[stim_nm]=np.zeros(n_electrodes)
        pvals_by_wavs[stim_nm]=np.zeros(n_electrodes)
        for ielec in range(n_electrodes):
            try:
                corr,pval=pearsonr(first_split_mean[:,ielec],second_split_mean[:,ielec])
            except:
                pass
            corrs_by_wavs[stim_nm][ielec]=corr
            pvals_by_wavs[stim_nm][ielec]=pval



    return corrs_by_wavs,pvals_by_wavs


#%% full script
if __name__=='__main__':

    all_subj_data=load_all_subj_data()
    fs_trf=100 # evnt times in seconds; use trf sampling rate for preprocessed data
    sample_subjs=[k for k in all_subj_data.keys()]
    eeg_by_wavs=split_eeg_trials_to_wavs(all_subj_data,fs=fs_trf)
    corrs_by_wavs,pvals_by_wavs=split_half_corr(eeg_by_wavs)
    

 

#OLD IDEA:
    
# STEP TWO: USE SEGMENT NAMES TO PARTITION DATA INTO TRIALS USING SET OPERATIONS
# MIGHT BE WORTHWHILE TO CHECK OUR ASSUMPTION ABOUT ALL TRIALS BELONGING TO THE SAME SET
# first, isolate names of wav files in each trial from all subjects 
    

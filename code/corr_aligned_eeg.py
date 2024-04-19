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
    subj_order_by_wavs={}
    all_subjs_ordered=[]
    for subj, subj_data in all_subj_data.items():
        print(f"loading timestamps for subject {subj}")
        evnt=utils.load_evnt(subj)
        evnt_nms,_,_,evnt_onsets,evnt_offsets=utils.extract_evnt_data(evnt,fs)
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
                    # stim already in dict from previous subject, add to existing list
                    eeg_by_wavs[nm].append(eeg[stim_trial_onset:stim_trial_offset])
                    subj_order_by_wavs[nm].append(subj)
                else:
                    # create new list for this stim
                    eeg_by_wavs[nm]=[eeg[stim_trial_onset:stim_trial_offset]]
                    subj_order_by_wavs[nm]=[subj]
            
        all_subjs_ordered.append(subj)
    return eeg_by_wavs,subj_order_by_wavs,all_subjs_ordered

# STEP 4: CALCULATE SPLIT-HALF CORRELATION
from scipy.stats import pearsonr
def split_half_corr_wavs_separate(eeg_by_wavs):
    '''
    calculates split half correlation for each individual wav, ignoring subjects without corresponding response and ignoring missing wavs
    '''
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

def concat_all_responses(eeg_by_wavs,subj_order_by_wavs,all_subjs_ordered):
    '''
    very inefficient way of organzing all subject responses to the same array with zeros where missing stims
    '''
    all_subj_data_concat={subj:[] for subj in all_subjs_ordered}
    _n_electrodes=62
    for stim_nm,all_subjs_eeg in eeg_by_wavs.items():
        subjs_with_stim=subj_order_by_wavs[stim_nm]
        # force to be same duration
        all_subjs_eeg_trimmed=trim_lengths_to_match(all_subjs_eeg_same_stim=all_subjs_eeg)
        resp_len=all_subjs_eeg_trimmed[0].shape[0]
        for subj in all_subjs_ordered:
            if subj in subjs_with_stim:
                # add subject's re
                # seems kinda stupid to do a list comprehension for a single index but idk what else to do here
                subj_idx=[idx for idx,n in enumerate(subjs_with_stim) if n==subj][0]
                subj_stim_resp=all_subjs_eeg_trimmed[subj_idx]
                all_subj_data_concat[subj].append(subj_stim_resp)
            else:
                all_subj_data_concat[subj].append(np.zeros((resp_len,_n_electrodes)))
    # once all responses organized in same place, turn lists to np arrays to concat
    for subj,eeg_lists in all_subj_data_concat:
        all_subj_data_concat[subj]=np.concatenate(eeg_lists)
    return all_subj_data_concat
from scipy.stats import pearsonr
def split_half_corr_concat(all_subj_data_concat):
    '''
    returns
    corrs_by_electrodes: np.array [electrodes x 2] where second dimension corresponds to p values
    '''
    _n_electrodes=62
    corrs_by_electrodes=np.zeros((_n_electrodes,2))
    subjs=[k for k in all_subj_data_concat]
    first_split_subjs=subjs[:len(subjs)//2]
    second_split_subjs=subjs[len(subjs)//2:]
    first_split_eeg=np.asarray([all_subj_data_concat[s] for s in first_split_subjs])
    second_split_eeg=np.asarray([all_subj_data_concat[s] for s in second_split_subjs])
    first_split_mean=first_split_eeg.mean(axis=0)
    second_split_mean=second_split_eeg.mean(axis=0)
    for ielec in range(_n_electrodes):
        corrs_by_electrodes[ielec]=pearsonr(first_split_mean[ielec],second_split_mean[ielec])
    return corrs_by_electrodes
        


    
#%% full script
if __name__=='__main__':

    all_subj_data=load_all_subj_data()
    fs_trf=100 # evnt times in seconds; use trf sampling rate for preprocessed data
    sample_subjs=[k for k in all_subj_data.keys()]
    eeg_by_wavs,subj_order_by_wavs,all_subjs_ordered=split_eeg_trials_to_wavs(all_subj_data,fs=fs_trf)
    # corrs_by_wavs,pvals_by_wavs=split_half_corr_wavs_separate(eeg_by_wavs)
    del all_subj_data
    all_subj_data_concat=concat_all_responses(eeg_by_wavs,subj_order_by_wavs,all_subjs_ordered)
    corrs_by_electrodes=split_half_corr_concat(all_subj_data_concat)
    

 

#OLD IDEA:
    
# STEP TWO: USE SEGMENT NAMES TO PARTITION DATA INTO TRIALS USING SET OPERATIONS
# MIGHT BE WORTHWHILE TO CHECK OUR ASSUMPTION ABOUT ALL TRIALS BELONGING TO THE SAME SET
# first, isolate names of wav files in each trial from all subjects 
    

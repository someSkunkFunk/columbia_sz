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
    thresh_dir=f"thresh_{evnt_thresh}" # should be three digit number representing decimals to third place
    prep_data_dir=os.path.join("..","eeg_data","preprocessed_evnt",thresh_dir)
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
fs_trf=100 # evnt times in seconds; use trf sampling rate for preprocessed data
sample_subj='3328'
evnt=utils.load_evnt(sample_subj)
evnt_nms,evnt_blocks,evnt_confidence,evnt_onsets,evnt_offsets=utils.extract_evnt_data(evnt,fs_trf)

# STEP 3: GROUP TIMESTAMPS INTO TRIALS BASED ON PAUSES
# then recalculate onsets/offsets relative to start of trial?
def

# STEP 4: G



#OLD IDEA:
    
# STEP TWO: USE SEGMENT NAMES TO PARTITION DATA INTO TRIALS USING SET OPERATIONS
# MIGHT BE WORTHWHILE TO CHECK OUR ASSUMPTION ABOUT ALL TRIALS BELONGING TO THE SAME SET
# first, isolate names of wav files in each trial from all subjects 
    

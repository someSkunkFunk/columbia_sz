# %%
import scipy.io as spio
import numpy as np
import os
import sounddevice as sd
# %% LOAD/FORMAT STIM DATA
choose_stim_fl = 1 # 1 or 2
stim_fnm1 = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\Matlab_scripts EEG\EEG_Study\master_stim_file_schiz_study_old.mat"
stim_fnm2 = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\Matlab_scripts EEG\EEG_Study\master_stim_file_schiz_studybis.mat"

stim_fls = {1: stim_fnm1, 2: stim_fnm2}

all_stims_mat = spio.loadmat(stim_fls[choose_stim_fl], squeeze_me=True)

# NOTE: I think "orig_clean" and "orig_noisy" are regular wavs and 
#   "aud_clean"/"aud_noisy" are 100-band spectrograms?
# according to master_docEEG spectrograms are at 100 Hz? 
all_stims = all_stims_mat['stim']
# convert structured array to a dict based on dtype names 
# (which correspond to matlab struct fields)
stims_dict = {field_nm : all_stims[field_nm][:] for field_nm in all_stims.dtype.names}

# %% LOAD/FORMAT EEG DATA
# NOTE: folder structure in eeg_dir: $SUBJ_NUM/$BLOCK_NUM/("analog", 
# "Stimulus")
# ACTAULLY not all subjects have the analog folders, not sure why, but some 
# (once again, not all) have an "Original" folder with the hdf5 for the
# what I believe is the truly original eeg data and some of those also
# have a mat file specifying the stim order (again, I think?)
# will continue trying to get trf's based on assumption that these are
# the raw data files and ignore subjects where they aren't present 



# %% find subjects with "good" raw data
def find_good_data(eeg_dir, need_stim_order=True):
    '''
    Only keep subj IDs if:
    - "original" directory exists
    - original folder contains hdf5s for 6 blocks
    - original folder contains stim order .mat files 
        - not sure if these are required though
    NOTE: it may be the case that some subjects don't have hdf5s and/or stimOrder
    .mats in the original folder but do have data as htk or stimorder .mat in 
    a block-specific sub-directory that may be useful in the future
    although per the master word doc it seems that those files are pre-processed
    '''
    n_blocks = 6 # there should be six blocks for each subject
    good_hc_subjs = [] # save tup of strings (subj_num, )
    good_sp_subjs = []
    # get all subject ids
    hc_dir = os.path.join(eeg_dir, "hc")
    sp_dir = os.path.join(eeg_dir, "sp")
    # leave out last dir ("files for Load_Data") since not necessary?
    hc_candidates = [nm for nm in os.listdir(hc_dir) 
                     if nm != "files for Load_Data"]
    sp_candidates = [nm for nm in os.listdir(sp_dir) 
                     if nm != "files for Load_Data"]
    # filter hc subjects
    for subj in hc_candidates:
        subj_dir = os.path.join(hc_dir, subj)
        if "original" not in os.listdir(subj_dir):
            continue
        og_subj_dir = os.path.join(subj_dir, "original")
        subj_fnms = os.listdir(og_subj_dir)
        #NOTE: should give empty list if none
        hf_fls = [hf_nm for hf_nm in subj_fnms if hf_nm.endswith('.hdf5')]
        mt_fls = [mt_nm for mt_nm in subj_fnms if mt_nm.endswith('.mat')]
        # stupid solution, hopefully dosn't break: just concat all the flnms
        # of particular file type and see if megastring contains
        # B1-6
        # print(subj_fnms)
        # print(subj, hf_fls)
        # raise NotImplementedError
        hdfstr = ''.join([f"B{ii}" for ii in range(1,n_blocks+1) 
                          if f"B{ii}" in ''.join(hf_fls)])
        matstr = ''.join([f"B{ii}" for ii in range(1,n_blocks+1)
                          if f"B{ii}" in ''.join(mt_fls)])
        good_hc_subjs.append((subj, hdfstr, matstr))

    # filter sp subjects
    for subj in sp_candidates:
        subj_dir = os.path.join(sp_dir, subj)
        if "original" not in os.listdir(subj_dir):
            continue
        og_subj_dir = os.path.join(subj_dir, "original")
        subj_fnms = os.listdir(og_subj_dir)
        #NOTE: should give empty list if none
        hf_fls = [hf_nm for hf_nm in subj_fnms if hf_nm.endswith('.hdf5')]
        mt_fls = [mt_nm for mt_nm in subj_fnms if mt_nm.endswith('.mat')]
        # stupid solution, hopefully dosn't break: just concat all the flnms
        # of particular file type and see if megastring contains
        # B1-6
        # print(subj_fnms)
        # print(subj, hf_fls)
        # raise NotImplementedError
        hdfstr = ''.join([f"B{ii}" for ii in range(1,n_blocks+1) 
                          if f"B{ii}" in ''.join(hf_fls)])
        matstr = ''.join([f"B{ii}" for ii in range(1,n_blocks+1)
                          if f"B{ii}" in ''.join(mt_fls)])
        good_sp_subjs.append((subj, hdfstr, matstr))

    return good_hc_subjs, good_sp_subjs

hc_keep, sp_keep = find_good_data()
# print(f"hc_keep:{hc_keep} \n sp_keep: {sp_keep}")
# %% look at individual subject's data
eeg_dir = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\EEG For Emeline\Schizophrenia EEG Study\Subject Data and Results"
choose_hc_sp = "hc" # healthy control ("hc") or schizophrenia patients ("sp")
subj_num = '3253' # arbitrarily choose one that has all eeg hdf5s and stim mats
n_blocks = 6
blocks = [f"B{ii}" for ii in range(1, n_blocks+1)]
         
# %% get current subject's eeg data
import h5py
eeg_fnms = [fnm for fnm in os.listdir(os.path.join(eeg_dir, choose_hc_sp, subj_num, "original")) if fnm.endswith('.hdf5')]
#NOTE: below assumes data for all six blocks present in original folder
eeg_fnms_dict = {block_num: fnm for block_num, fnm in zip(blocks, eeg_fnms)}
subj_eeg = {}
for block_num in blocks:
    # get eeg responses and stim order info
    eeg_fnm = os.path.join(eeg_dir, choose_hc_sp, subj_num, "original",
                           eeg_fnms_dict[block_num])
    block_file = h5py.File(eeg_fnm) #returns a file; read mode is default
    #NOTE: don't need to unpack stim order mats into dict since only one field
    # keys returned in each hdf5 file: ['AsynchronData', 'RawData', 'SavedFeatues', 'Version']
    # keys returned in subj_eeg['BX']['RawData']:  
    # ['AcquisitionTaskDescription', 'DAQDeviceCapabilities',
    #  'DAQDeviceDescription', 'Samples', 'SessionDescription', 'SubjectDescription']
    # not sure if any of this additional information is useful but for now just keeping samples 
    # since that seems to be the actual recording
    # subj_eeg['B1']['RawData']['Samples'].shape:
    # (2018496, 65) 65: number of channels???
    subj_eeg[block_num] =  np.asarray(block_file['RawData']['Samples'])
    del block_file
# get stim names/order for current subject
for ii, block_num in enumerate(blocks):
    
    stim_order_fnm = os.path.join(eeg_dir, choose_hc_sp, 
                                subj_num, "original",
                                "_".join([block_num, "stimorder.mat"]))

    if ii == 0:
        subj_stim_order = spio.loadmat(stim_order_fnm, squeeze_me=True)['StimOrder']
    else:
        block_stim_order_mat = spio.loadmat(stim_order_fnm, squeeze_me=True)['StimOrder']
        subj_stim_order = np.hstack((subj_stim_order, block_stim_order_mat))

# NOTE: '.wav' at end of ID in subj_stim_order won't match the actual stim ID
# create input matrix from stimuli subject has
# subj_X = [stims_dict]

#%% use subject's stimorder files to find start and stop times for each stimulus
# use noisy stim wav file for xcorr, but use clean for trf analysis
from scipy import signal

fs_audio = stims_dict['fs'][0] # 11025 originally
fs_eeg = 2400
for wav_nm in subj_stim_order:
    # loop through each stimulus
    idx = stims_dict['ID'] == wav_nm[:-4]
    # find wav for current stimulus
    noisy_wvform = stims_dict['orig_noisy'][idx][0]
    wav_dur = (fs_audio)**-1 * (noisy_wvform.size - 1) # duration in s
    b, a = signal.butter(8, fs_eeg/4, fs=fs_audio)
    noisy_wvform = signal.filtfilt(b, a, noisy_wvform) # antialias filt
    noisy_wvform = signal.resample(noisy_wvform, np.floor(wav_dur*fs_eeg)) # ds
    #TODO: doesn't make sense to loop over each block here again but just
    # want to try the correlation to see if it will work
    for block in blocks:
        # TODO: downsample waveform too fs_eeg
        # TODO: correlate ds waveform w eeg audio
        # TODO: how to get start and stop times from cross correlation?
        # TODO: use prior stimuli end time as starting point for next stim search
        
        pass
#%%

sound_frag = subj_eeg['B1'][:,-1]
sd.play(sound_frag, fs_eeg)

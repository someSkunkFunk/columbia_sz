# same preprocessing code as preprocess_segment, but use evnt timestamps for segmenting, assuming a shift of 12000 samples
# for subjects without timestamps, first get the timestamps, then preprocess and segment
#%%
# imports, etc.

import pickle
import scipy.io as spio
import numpy as np
import os
# import sounddevice as sd # note not available via conda....? not sure I'll need anyway so ignore for now
import h5py
from scipy import signal

import matplotlib.pyplot as plt
import utils

# specify fl paths assumes running from code as pwd
eeg_dir=os.path.join("..", "eeg_data")
# stim_fnm = "master_stim_file_schiz_studybis.mat" # note this is original fnm from box, we changed to just stim_info.mat
stim_fnm="stim_info.mat"
stim_fl_path=os.path.join(eeg_dir, "stim_info.mat")
stims_dict=utils.get_stims_dict(stim_fl_path)
fs_audio=stims_dict['fs'][0] # 11025 foriginally
fs_eeg=2400


n_blocks = 6
blocks = [f"B{ii}" for ii in range(1, n_blocks+1)]



#%%
# setup
filt_band_lims = [1.0, 15] #Hz; highpass, lowpass cutoffs
filt_o = 3 # order of filter (effective order x2 of this since using zero-phase)
fs_trf = 100 # Hz, downsample frequency for trf analysis
processed_dir_path=os.path.join(eeg_dir, "preprocessed2") #directory where processed data goes
#NOTE: need to uncomment bottom line when running slurm job via bash, commented for debugging purposes
subj_num=os.environ["subj_num"] #TODO: make bash go thru list of all subjects
subj_cat=utils.get_subj_cat(subj_num) #note: checked get_subj_cat, should be fine
raw_dir=os.path.join(eeg_dir, "raw")
print(f"Fetching data for {subj_num, subj_cat}")
subj_eeg = utils.get_full_raw_eeg(raw_dir, subj_cat, subj_num, blocks=blocks)

#%% 
# load evnt timestamps
#TODO: make evnt timestamps loading function a util
# check if save directory exists, else make one
save_path = os.path.join(processed_dir_path, subj_cat, subj_num)
if not os.path.isdir(save_path):
    print(f"preprocessed dir for {subj_num, subj_cat} not found, creating one.")
    os.makedirs(save_path, exist_ok=True)
# find evnt timestamps
timestamps_path=os.path.join("..","eeg_data","timestamps",f"evnt_{subj_num}.mat")
evnt_mat=spio.loadmat(timestamps_path)
# returns dict for some reason, which mat2dict doesnt like
evnt=evnt_mat['evnt']
evnt=utils.mat2dict(evnt)

#%%
# preprocess each block separately
shift=12000 #empirical shift between their timestamps and mine in samples 
confidence_thresh=0.4
# (for first stim, assuming consistent across stims)
timestamps_ds = {}
print(f"Start: preprocessing for {subj_num, subj_cat}")
for block, raw_eeg in subj_eeg.items():
    print(f"start: {block}")
    timestamps_ds[block] = {}
    # filter and resample
    if fs_eeg / 2 <= fs_trf:
        raise NotImplementedError("Nyquist") 
    sos = signal.butter(filt_o, filt_band_lims, btype='bandpass', output='sos', fs=fs_eeg)
    raw_eeg = signal.sosfiltfilt(sos, raw_eeg, axis=0)
    ds_factor = int(np.floor((raw_eeg.shape[0]-1)*(fs_trf/fs_eeg)))
    subj_eeg[block] = signal.resample(raw_eeg, ds_factor, axis=0)
    # resample timestamps
    for stim_nm in evnt['name'][0]:
        stim_nm=stim_nm[0] # weird array indexing is stupid
        stim_index=evnt['name'][0]==stim_nm
        start=evnt['syncPosition'][0,stim_index][0][0][0]+shift
        
        #TODO:FIND END, IMPLEMENT CONFIDENCE-BASED REJECTION
        stim_wav=utils.get_stim_wav(stims_dict,stim_nm,'clean')
        stim_dur=(stim_wav.size-1)/fs_audio
        end=start+int(stim_dur*fs_eeg+1)
        confidence=evnt['confidence'][0,stim_index][0][0][0]
        
        if confidence > confidence_thresh:#TODO: HOW THE FUCK DO WE KNOW WHEN TO LEAVE OUT EVNT TIMESTAMPS?
            # start/end timestamps after downsampling
            s_ds = int(np.floor(start*(fs_trf/fs_eeg)))
            e_ds = int(np.floor(end*(fs_trf/fs_eeg))) #NOTE: off by one error maybe?
            timestamps_ds[block][stim_nm] = (s_ds, e_ds, confidence)
        else:
            timestamps_ds[block][stim_nm] = (None, None, confidence)
#%%
# align downsampled eeg using ds timestamps
print(f"Preprocessing done for {subj_num, subj_cat}. algining and slicing eeg")
#TODO: resulting pkl file has nothing... probably align responses doing something weird?
subj_data = utils.align_responses(subj_eeg, timestamps_ds, stims_dict)
subj_data['fs'] = fs_trf
print("subj_data before pickling:")
print(subj_data.head())
print(f'saving to: {save_path}')
subj_data.to_pickle(os.path.join(save_path, "aligned_resp.pkl"))
print(f"{subj_num, subj_cat} preprocessing and segmentation complete.")
# break


 
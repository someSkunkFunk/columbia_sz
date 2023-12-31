# Already have stim timestamps for each stim in some subjects; for those
# we just preprocess and segment
# for subjects without timestamps, first get the timestamps, then preprocess and segment
#%%
# imports

import pickle
import scipy.io as spio
import numpy as np
import os
# import sounddevice as sd # note not available via conda....? not sure I'll need anyway so ignore for now
import h5py
from scipy import signal

import matplotlib.pyplot as plt
import utils
#%% 
# specify fl paths assumes running from code as pwd
eeg_dir =os.path.join("..", "eeg_data")
# stim_fnm = "master_stim_file_schiz_studybis.mat" # note this is original fnm from box, we changed to just stim_info.mat
stim_fnm="stim_info.mat"
stim_fl_path=os.path.join(eeg_dir, "stim_info.mat")
stims_dict = utils.get_stims_dict(stim_fl_path)
fs_audio = stims_dict['fs'][0] # 11025 foriginally
fs_eeg = 2400


n_blocks = 6
blocks = [f"B{ii}" for ii in range(1, n_blocks+1)]



#%%
# setup
filt_band_lims = [1.0, 15] #Hz; highpass, lowpass cutoffs
filt_o = 3 # order of filter (effective order x2 of this since using zero-phase)
fs_trf = 100 # Hz, downsample frequency for trf analysis
processed_dir_path=os.path.join(eeg_dir, "preprocessed") #directory where processed data goes
#NOTE: need to uncomment bottom line when running slurm job via bash, commented for debugging purposes
# subj_num=os.environ["subj_num"] #TODO: make bash go thru list of all subjects
subj_cat=utils.get_subj_cat(subj_num) #note: checked get_subj_cat, should be fine
raw_dir=os.path.join(eeg_dir, "raw")
subj_eeg = utils.get_full_raw_eeg(raw_dir, subj_cat, subj_num, blocks=blocks)

#%% 
# Find timestamps
save_path = os.path.join(processed_dir_path, subj_cat, subj_num)
# check if directory exists, else make one
if not os.path.isdir(save_path):
    print(f"preprocessed dir for {subj_num, subj_cat} not found, creating one.")
    os.makedirs(save_path, exist_ok=True)
# check if timestamps fl exists already
timestamps_path = os.path.join(save_path, f"{subj_num}_timestamps.pkl")
if os.path.exists(timestamps_path):
    # if already have timestamps, load from pkl:
    print(f"{subj_num, subj_cat} already has timestamps, loading from pkl.")
    with open(timestamps_path, 'rb') as pkl_fl: 
        timestamps = pickle.load(pkl_fl)
else:
    print(f"Generating timestamps for {subj_num, subj_cat} ...")
    # get timestamps
    timestamps = utils.get_timestamps(subj_eeg, raw_dir, subj_num, subj_cat, stims_dict, blocks)
    #  save stim timestamps
    with open(os.path.join(save_path, subj_num+'_timestamps.pkl'), 'wb') as f:
        pickle.dump(timestamps, f)
#%%
# preprocess each block separately

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
    for stim_nm, (start, end, confidence) in timestamps[block].items():
        if start is not None:
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
subj_data.to_pickle(os.path.join(save_path, "aligned_resp.pkl"))
print(f"{subj_num, subj_cat} preprocessing and segmentation complete.")
# break


 
            

        
# %%
# check result for one subject
#NOTE: visually not very convincing but we had a shitty audio to begin with, then filtered and downsampled it so....
subj_num = "3253"
subj_cat = "hc"
with open(os.path.join(os.getcwd(), subj_cat, subj_num, "aligned_resp.pkl"), 'rb') as pkl_fl:
    subj_data = pickle.load(pkl_fl)


#%%
# sort stims by stim align confidence vals
subj_data.sort_values(by='confidence', ascending=True, inplace=True) # sort by confidence vals
#NOTE: ALL CODE BELOW ASSUMES subj_data RESORTED BY CONFIDENCE VALS
#TODO: probably better to have consistent stim ordering across subjects rather than this
#%%
# filter out nans and check if eeg recording duration is always larger than stim duration
stim_durs = [(stims_dict['orig_noisy'][stims_dict['ID']==s_nm[:-4]][0].size - 1)/fs_audio for s_nm in subj_data.loc[subj_data['eeg'].notna(), 'stim_nms']]
response_durs = [(r.size - 1)/fs_eeg for r in subj_data.loc[subj_data['eeg'].notna(), 'eeg_audio']]
print(np.all(response_durs < stim_durs))
#NOTE: so all responses consistently shorter than stims due to stim downsampling during xcorr process
# going to just downsample all the stims then pad responses at the beginning if needed to match stim size (shouldn't need)
#%%
# sort by confidence and visually compare
s_num = 700 # which to plot, in order of confidence
subj_data.sort_values(by='confidence', ascending=True, inplace=True)
stim_nm = subj_data.iloc[s_num]['stim_nms']
stim = utils.get_stim_wav(stims_dict, stim_nm, has_wav=True) # gets noisy by default
t_stim = utils.make_time_vector(fs_audio, stim.size)
recording = subj_data.iloc[s_num]['eeg_audio']
if np.isnan(recording).any():
    print(f"recording is {recording}")
else:
    t_recording = utils.make_time_vector(fs_eeg, recording.size)
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(t_stim, stim)
    ax[0].set_title('OG Stim')
    ax[1].plot(t_recording, recording, label=f'confidence:{subj_data.iloc[s_num].confidence}')
    ax[1].set_title('Recorded audio')
    ax[1].set_xlabel('Seconds')
    plt.legend()
    fig.suptitle(f'{subj_data.iloc[s_num].stim_nms}')
    
#%%
# listen to stimulus
sd.play(stim, fs_audio)
#%%
# listen to recording (from df)
sd.play(recording, fs_trf)

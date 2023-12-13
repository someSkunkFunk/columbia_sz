# segment eeg audio recording via envelope thresholding
# %%
import scipy.io as spio
import numpy as np
import os
import sounddevice as sd
import h5py
from scipy import signal
import matplotlib.pyplot as plt
#%% 
# Get stim information file
stim_fnm1 = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\Matlab_scripts EEG\EEG_Study\master_stim_file_schiz_study_old.mat"
# NOTE: determined that both stim files have same orig_noisy stim wavs

all_stims_mat = spio.loadmat(stim_fnm1, squeeze_me=True)

# NOTE: I think "orig_clean" and "orig_noisy" are regular wavs and 
#   "aud_clean"/"aud_noisy" are 100-band spectrograms?
# according to master_docEEG spectrograms are at 100 Hz? 
all_stims = all_stims_mat['stim']
# convert structured array to a dict based on dtype names 
# (which correspond to matlab struct fields)
stims_dict = {field_nm : all_stims[field_nm][:] for field_nm in all_stims.dtype.names}
#NOTE: stim durations between 0.87s and 10.6 s
# %% 
# Select Subject
eeg_dir = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\EEG For Emeline\Schizophrenia EEG Study\Subject Data and Results"
choose_hc_sp = "hc" # healthy control ("hc") or schizophrenia patients ("sp")
subj_num = '3316' # has all eeg hdf5s and stim mats AND evnt structure for comparison
n_blocks = 6
blocks = [f"B{ii}" for ii in range(1, n_blocks+1)]

fs_audio = stims_dict['fs'][0] # 11025 originally
fs_eeg = 2400

eeg_fnms = [fnm for fnm in os.listdir(os.path.join(eeg_dir, choose_hc_sp, subj_num, "original")) if fnm.endswith('.hdf5')]
#NOTE: below assumes data for all six blocks present in original folder
eeg_fnms_dict = {block_num: fnm for block_num, fnm in zip(blocks, eeg_fnms)}
subj_eeg = {}
# %%
# get eeg data, calculate envelopes
crit_freq = 0.1 # cutoff freq for filters
thresh = 310 #threshold for lowpassed envelope

for block_num in blocks:
    # get eeg 
    eeg_fnm = os.path.join(eeg_dir, choose_hc_sp, subj_num, "original",
                           eeg_fnms_dict[block_num])
    block_file = h5py.File(eeg_fnm) #returns a file; read mode is default
    subj_eeg[block_num] =  np.asarray(block_file['RawData']['Samples'])
    del block_file
    # get stim order file
    stim_order_fnm = os.path.join(eeg_dir, choose_hc_sp, 
                                subj_num, "original",
                                "_".join([block_num, "stimorder.mat"])
                                )
    block_stim_order = spio.loadmat(stim_order_fnm, squeeze_me=True)['StimOrder']
    
    recording_envelope = np.abs(signal.hilbert(subj_eeg[block_num][:,-1]))
    # create lowpass filter
    sos = signal.butter(16, crit_freq, fs=fs_eeg, output='sos')
    lp_env = signal.sosfiltfilt(sos, recording_envelope)
    t = np.arange(0, lp_env.size/fs_eeg, 1/fs_eeg)
    on_off_times = np.zeros_like(t, dtype=bool)
    on_off_times[1:] = np.diff(lp_env>thresh)
    ylim = [0, 1000]
    plt.plot(t, recording_envelope)
    plt.plot(t, lp_env)
    plt.xlabel('seconds')
    plt.hlines(thresh, t[0], t[-1], color='r')
    plt.vlines(t[on_off_times], ylim[0], ylim[1], color='r')
    plt.ylim(ylim)
    
    raise NotImplementedError
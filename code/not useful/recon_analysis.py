# %%
import scipy.io as spio
import numpy as np
import os
import sounddevice as sd
import h5py
from scipy import signal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import utils
import mtrf
#%% 
# Get stim information file and subject eeg
stim_fnm = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\Matlab_scripts EEG\EEG_Study\master_stim_file_schiz_study_old.mat"
stims_dict = utils.get_stims_dict(stim_fnm)
# Select Subject, get full eeg dataset
eeg_dir = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\EEG For Emeline\Schizophrenia EEG Study\Subject Data and Results"
choose_hc_sp = "hc" # healthy control ("hc") or schizophrenia patients ("sp")
subj_num = '3316' # has all eeg hdf5s and stim mats AND evnt structure for comparison
n_blocks = 6
blocks = [f"B{ii}" for ii in range(1, n_blocks+1)]


subj_eeg = utils.get_full_raw_eeg(eeg_dir, choose_hc_sp, subj_num, blocks=blocks)
fs_audio = stims_dict['fs'][0]
fs_eeg = 2400
#%%
# load stimuli timing info
event_fnm = os.path.join(eeg_dir, choose_hc_sp, subj_num, "evnt_3316.mat")
event_mat = spio.loadmat(event_fnm, squeeze_me=True)['evnt']
event_dict = utils.mat2dict(event_mat)
#%%
# segment raw eeg based on event info
#TODO: probably want some data before/after stimulus?
# save stimuli and responses
# S = []
# R = []
# names = []
# for block in blocks:
#     block_idx = event_dict['block'] == block
#     block_stim_nms = event_dict['name'][block_idx]
#     for nm in block_stim_nms:
#         # get original clean stimulus
#         stim_idx = stims_dict['ID'] == nm[:-4]
#         stim = stims_dict['orig_clean'][stim_idx][0]
#         # downsample stim and add to S
#         sos = signal.butter(16, fs_eeg/3, fs = fs_audio, output='sos')
#         stim = signal.sosfiltfilt(sos, stim)
#         stim = signal.resample(stim, int(stim.size*fs_eeg/fs_audio))
#         S.append(stim)
#         # get segment of subj_eeg corresponding to that stim
#         start = int(event_dict['startTime'][event_dict['name']==nm]*fs_eeg)
#         end = int(event_dict['stopTime'][event_dict['name']==nm]*fs_eeg)
#         R.append(subj_eeg[block][start:end+1])
#         names.append(nm)
#NOTE: seems start/stop times in event mat already have some pre/post stim time as well
#%%
# downsample stimuli
S = []
for s in S_clean:
    s= s[0]
    sos = signal.butter(16, fs_eeg/3, fs = fs_audio, output='sos')
    s = signal.sosfiltfilt(sos, s)
    s = signal.resample(s, int(s.size*fs_eeg/fs_audio))
    S.append(s)

#%%
from mtrf.model import TRF
# do backwards trf, reconstruct stim envelope
# NOTE: only first channels 63/64 are reference electrodes and 65 is audio recording
envelopes = [np.abs(signal.hilbert(s)) for s in S]
responses = [r for r in R]
#TODO: need to pad stimuli with zeros before and after... to match responses...:{}
decoder = TRF(direction=-1)
tmin, tmax = -0.2, 0.5 # in s
decoder.train(responses, envelopes, fs_eeg, tmin, tmax, regularization=100)
r, mse = mtrf.stats.cross_validate(decoder, envelopes, responses)
# decoder.train(envelopes, responses, fs_eeg)

#%%
# random debugging
# NOTE: seems like the event array sliced a much longer timeframe than original stim?
# stim1_dur = (event_dict['stopTime'][0] - event_dict['startTime'][0])
# stim1_dur_og = (stims_dict['orig_noisy'][stims_dict['ID']==event_dict['name'][0][:-4]][0].size -1)/fs_audio
# # %%
# # code for comparing downsampled stimuli with resulting eeg recording slices:
# stim_num = -40
# stim_nm = event_dict['name'][stim_num]
# labels = ['og', 'recorded']

# fig, ax = utils.plot_waveform([S[stim_num], R[stim_num][:,-1]], fs_eeg, labels=labels, nrows=2)
# [axes.set_xlabel('Seconds') for axes in ax]
# fig.suptitle(' '.join([subj_num, stim_nm[:-4]]))
# plt.show()
# determine how different the two stim files are
# result: both stim info mat files contain the same wav files
# %%
import scipy.io as spio
import numpy as np
import os
import sounddevice as sd
import h5py
from scipy import signal
#%% 
# Get stim information file NOTE: not sure which of these is right/if different?
stim_fnm1 = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\Matlab_scripts EEG\EEG_Study\master_stim_file_schiz_study_old.mat"
stim_fnm2 = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\Matlab_scripts EEG\EEG_Study\master_stim_file_schiz_studybis.mat"

stim_fls = {1: stim_fnm1, 2: stim_fnm2}

all_stims_mat1 = spio.loadmat(stim_fls[1], squeeze_me=True)
all_stims_mat2 = spio.loadmat(stim_fls[2], squeeze_me=True)

all_stims1 = all_stims_mat1['stim']
all_stims2 = all_stims_mat2['stim']
# convert structured array to a dict based on dtype names 
# (which correspond to matlab struct fields)
stims_dict1 = {field_nm : all_stims1[field_nm][:] for field_nm in all_stims1.dtype.names}
stims_dict2 = {field_nm : all_stims2[field_nm][:] for field_nm in all_stims2.dtype.names}
n_stims = 784
comps = np.ones(n_stims, dtype=bool)
for ii in range(n_stims):
    comps[ii] == np.array_equal(stims_dict1['orig_noisy'][ii], stims_dict2['orig_noisy'][ii])    

np.all(comps) # returns True, so shouldn't matter
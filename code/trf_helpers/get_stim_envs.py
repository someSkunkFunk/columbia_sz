from scipy import signal
import numpy as np
def get_stim_envs(stims_dict, clean_or_noisy, fs_output, f_lp=30):
    ''''
    stims_dict: classic stims dict with both dirty and clean stims from stim file
    clean_or_noisy: choose which stims to downsample and extract envelopes from "clean" or "noisy"
    fs_output: desired sampling frequency for resulting envelopes
    f_lp: low-pass filter cutoff, 30 Hz by default (must be below fs_output/2)
    returns: stim envelopes sampled at fs_output
    '''
    if f_lp >= fs_output/2:
        raise NotImplementedError('low-pass is not below nyquist')
    fs_stim = stims_dict['fs'][0]
    sos = signal.butter(3, f_lp, fs = fs_stim, output='sos')
    # get envelopes first, then resample
    stim_envs = {stim_nm: np.abs(signal.hilbert(stim_wav)) 
                 for stim_nm, stim_wav in zip(stims_dict['ID'], stims_dict["orig_"+clean_or_noisy])}


    ds_envs = {stim_nm: signal.resample(signal.sosfiltfilt(sos, stim_wav), int((stim_wav.size - 1)*fs_output/fs_stim)) 
                for stim_nm, stim_wav in stim_envs.items()}
    # eliminate any negative values
    ds_envs = {stim_nm: np.maximum(s, 0) for stim_nm, s in ds_envs.items()}
    
    return ds_envs
import numpy as np
import pandas as pd
def align_responses(subj_eeg:dict, stim_start_end:dict, stims_dict:dict):
    '''
    subj_eeg: {'block_num': [time x channels] - eeg numpy array}
    stim_start_end {'block_num': {'stim_nm': (onset_time, offset_time)} }
    stims_dict: master stim directory with original stim wav files (noisy and clean)
    '''
    #TODO: add optional padding around on/of time edges 
    # "unwrapping" blocks dim
    

    S_names = []
    # timestamps = []
    confidence_vals = []
    for block in subj_eeg:
        for stim_nm, stim_info in stim_start_end[block].items():
            S_names.append(stim_nm)
            # timestamps.append(stim_info[:2])
            confidence_vals.append(stim_info[2])
    S_names = np.array(S_names)
    # timestamps = np.array(timestamps)
    confidence_vals = np.array(confidence_vals)


    # select stim slices from eeg
    R =[]
    R_audio = []

    for block in subj_eeg:
        for stim_nm, (start, end, _) in stim_start_end[block].items():
            if start is not None:
                R.append(subj_eeg[block][start:end,:62])
                R_audio.append(subj_eeg[block][start:end,-1])
            else:
                R.append(np.nan)
                R_audio.append(np.nan)
    # TODO: make this the default and without extraneous loops
    
    subj_data = pd.DataFrame(data={
        'stim_nms': S_names,
        'eeg': R,
        'eeg_audio': R_audio,
        'confidence': confidence_vals
    })
    
    return subj_data

import os
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
        if not any(mt_fls) and need_stim_order:
            # skip subjects with no stimorder files
            continue
        # stupid solution, hopefully dosn't break: just concat all the flnms
        # of particular file type and see if megastring contains
        # B1-6

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
        if not any(mt_fls) and need_stim_order:
            # skip subjects with no stimorder files
            continue
        # stupid solution, hopefully dosn't break: just concat all the flnms
        # of particular file type and see if megastring contains

        hdfstr = ''.join([f"B{ii}" for ii in range(1,n_blocks+1) 
                          if f"B{ii}" in ''.join(hf_fls)])
        matstr = ''.join([f"B{ii}" for ii in range(1,n_blocks+1)
                          if f"B{ii}" in ''.join(mt_fls)])
        good_sp_subjs.append((subj, hdfstr, matstr))

    return good_hc_subjs, good_sp_subjs

import os
import h5py
import numpy as np
def get_full_raw_eeg(raw_dir, subj_cat:str, subj_num:str, blocks=None):
    
    """
    raw_dir: root directory where raw eeg data is stored
    subj_cat: ("hc" or "sp") sub-directory for healthy control subjects or schizophrenia patients
    subj_num: individual subject number
    blocks: which blocks to get for that subject
    returns:
        subj_eeg: {'block_num': np.ndarray (time x channels) }
    """
    if blocks is None:
        # get all 6 blocks by default
        blocks = [f"B{ii}" for ii in range(1, 6+1)]
    # get raw data hdf5 fnms
    eeg_fnms = [fnm for fnm in os.listdir(os.path.join(raw_dir, subj_cat, subj_num, "original")) if fnm.endswith('.hdf5')]
    #NOTE: below assumes data for all six blocks present in original folder
    #NOTE: also assumes that block names in order.. for subj 3244 B1 was not present but eeg_fnms_dict assigned keys w block names
    # that were off by one because of it...
    eeg_fnms_dict = {block_num: fnm for block_num, fnm in zip(blocks, eeg_fnms)}
    subj_eeg = {}
    # get raw eeg data

    for block_num in blocks:

        # get eeg 
        eeg_fnm = os.path.join(raw_dir, subj_cat, subj_num, "original",
                            eeg_fnms_dict[block_num])
        block_file = h5py.File(eeg_fnm) #returns a file; read mode is default
        subj_eeg[block_num] =  np.asarray(block_file['RawData']['Samples'])
        del block_file
    return subj_eeg

from scipy import signal
def get_lp_env(env, crit_freq, fs):
    '''
    env: recording envelope
    crit_freq: lowpass filter cutoff f
    fs: recording fs
    '''
    # create lowpass filter
    sos = signal.butter(16, crit_freq, fs=fs, output='sos')
    lp_env = signal.sosfiltfilt(sos, env)
    return lp_env

def get_missing_stim_nms(timestamps):
    '''
    timestamps: {'block': {'stim_nm': (start, end, confidence)}}
    '''
    missing_stims = []
    for block in timestamps.keys():
        for stim_nm, (start, end, _) in timestamps[block].items():
            if start is not None:
                missing_stims.append(stim_nm)
    return missing_stims

def get_stim_wav(stims_dict, stim_nm, has_wav:bool, noisy_or_clean='noisy'):
    if has_wav:
        # remove '.wav' from name to match stim mat file
        stim_indx = stims_dict['ID'] == stim_nm[:-4]
    else:
        stim_indx = stims_dict['ID'] == stim_nm
    
    if noisy_or_clean == 'noisy':
        return stims_dict['orig_noisy'][stim_indx][0]
    elif noisy_or_clean == 'clean':
        return stims_dict['orig_clean'][stim_indx][0]

import scipy.io as spio
# from .mat2dict import mat2dict
def get_stims_dict(stim_fnm):
    '''
    transform mat file to dictionary for 
    '''
    # NOTE: determined that both stim files have same orig_noisy stim wavs
    
    all_stims_mat = spio.loadmat(stim_fnm, squeeze_me=True)

    # NOTE: I think "orig_clean" and "orig_noisy" are regular wavs and 
    #   "aud_clean"/"aud_noisy" are 100-band spectrograms?
    # according to master_docEEG spectrograms are at 100 Hz? 
    all_stims = all_stims_mat['stim']
    # convert structured array to a dict based on dtype names 

    # (which correspond to matlab struct fields)
    stims_dict = mat2dict(all_stims)
    #NOTE: stim durations between 0.87s and 10.6 s
    return stims_dict

import os
def get_subj_cat(subj_num, eeg_dir=None):
    """
    create dictionary for looking up subject number based on healthy control (hc) or 
    schizophrenia patient (sp) categories
    """
    if eeg_dir is None:
        # lookup based on folder structure of eeg_dir, default:
        eeg_dir = os.path.join( '..', "eeg_data")
    hc_folder = os.path.join(eeg_dir, "raw", "hc")
    sp_folder = os.path.join(eeg_dir, "raw","sp")

    # print("eeg_dir: ", eeg_dir)
    if os.path.exists(os.path.join(hc_folder, subj_num)):
        return "hc"
    elif os.path.exists(os.path.join(sp_folder, subj_num)):
        return "sp"
    else:
        raise NotImplementedError("subj category could not be found.")
 
def get_time_between_stims(stim_start_end, fs):
    times_between = {}
    for block in stim_start_end.keys():
        times_between[block] = {} 
        #NOTE: I believe start/end are sample indices but should double check
        prev_end = None
        prev_stim_nm = None
        for stim_nm, (start, end, _) in stim_start_end[block].items():
            if start is not None:
                if prev_end is None:
                    # first stim
                    prev_end = end
                    prev_stim_nm = stim_nm
                else:
                    # record time difference in samples and time
                    trans_nm = prev_stim_nm+stim_nm
                    times_between[block][trans_nm] = ( int(start-prev_end), (start - prev_end)/fs)
                    prev_end = end
                    prev_stim_nm = stim_nm
            else:
                # skip missing stims
                continue 
    return times_between

import numpy as np
import scipy.io as spio
import os
from scipy import signal
# NOTE: don't need imports from utils anymore since on same module?
# from utils import get_lp_env, set_thresh, segment, get_stim_wav, match_waves    
def get_timestamps(subj_eeg, eeg_dir, subj_num, subj_cat, stims_dict, blocks):
    fs_audio = stims_dict['fs'][0] # 11025 foriginally #TODO: UNHARDCODE
    fs_eeg = 2400 #TODO: UNHARDCODE
    which_corr = 'wavs' # 'wavs' or 'envs'
    # step size for xcorr window calculation
    confidence_lims = [0.90, 0.4] #max, min pearsonr correlation thresholds
    lpf_cut = 0.1 # lowpass cutoff for envelope-based segmetation

    # store indices for each block {"block_num":{"stim_nm": (start, end, rconfidence)}}
    timestamps = {}
    for block_num in blocks:
        timestamps[block_num] = {}
        # get stim order file
        stim_order_fnm = os.path.join(eeg_dir, subj_cat, 
                                    subj_num, "original",
                                    "_".join([block_num, "stimorder.mat"])
                                    )
        block_stim_order = spio.loadmat(stim_order_fnm, squeeze_me=True)['StimOrder']
        # get recording envelope
        rec_wav = subj_eeg[block_num][:,-1]
        rec_env = np.abs(signal.hilbert(rec_wav))

        if which_corr.lower() == 'envs':
            x = rec_env
        elif which_corr.lower() =='wavs':
            x = rec_wav
        else:
            raise NotImplementedError(f"which_corr = {which_corr} is not an option")
        prev_end=0 #TODO: verify this doesn't introduce new error (check w finished subject?)
        for stim_ii, stim_nm in enumerate(block_stim_order):
            print(f"processing stim {stim_ii+1} of {block_stim_order.size}")
            # grab stim wav
            stim_wav_og = get_stim_wav(stims_dict, stim_nm, has_wav=True)
            stim_dur = (stim_wav_og.size - 1)/fs_audio
            #TODO: what if window only gets part of stim? 
            # apply antialiasing filter to stim wav and get envelope  
            sos = signal.butter(8, fs_eeg/3, fs=fs_audio, output='sos')
            stim_wav = signal.sosfiltfilt(sos, stim_wav_og)
            stim_env = np.abs(signal.hilbert(stim_wav))
            # downsample envelope/wav depending on which is selected for wave matching
            if which_corr.lower() == 'wavs':
                # will "undersample" since sampling frequencies not perfect ratio
                y = signal.resample(stim_wav, int(np.floor(stim_dur*fs_eeg))) 
            elif which_corr.lower() == 'envs':
                # will "undersample" since sampling frequencies not perfect ratio
                y = signal.resample(stim_env, int(np.floor(stim_dur*fs_eeg))) 
                
            if (x.size < stim_wav.size):
                timestamps[block_num][stim_nm] = (None, None, None)
                continue

            print("checkpoint: matching waves")
            stim_on_off, confidence_val = match_waves(x, y, confidence_lims, fs_eeg)
            print("checkpoint: waves matched")
            if stim_on_off is not None:

                stim_start = np.where(stim_on_off)[0][0] + prev_end
                stim_end = np.where(stim_on_off)[0][1] + prev_end
                # save indices and confidence 
                timestamps[block_num][stim_nm] = (stim_start, stim_end, confidence_val)
                # don't look at recording up to this point again
                if which_corr.lower() == 'envs':
                    x = rec_env[stim_end:]
                elif which_corr.lower() =='wavs':
                    x = rec_wav[stim_end:]
                # mark end time of last stim found
                # NOTE: not sure if this will work if first stim in block can't be found
                prev_end = stim_end 
                # move onto next stim
                continue
            else:
                # missing stims
                timestamps[block_num][stim_nm] = (None, None, confidence_val)
                # move onto next stim
                continue
    return timestamps

def mat2dict(mat_array):
    '''
    reformat object numpy array returned by scipy.io.loadmat for mat files as dictionaries
    '''
    mat_dict = {field_nm: mat_array[field_nm][:] for field_nm in mat_array.dtype.names}
    return mat_dict

import numpy as np
def make_time_vector(fs, nsamples, start_time=0):
    return np.arange(start_time, nsamples/fs, 1/fs)

import os
import pickle
# don't need next line?
# from .get_subj_cat import get_subj_cat
def load_subj_data(subj_num, eeg_dir=None):
    '''
    helper to load subject data as pandas dataframe (pkl fl)
    '''
    subj_cat = get_subj_cat(subj_num)
    if eeg_dir is None:
        eeg_dir = os.path.join(os.getcwd(), '..', "eeg_data")
    subj_data_fnm = subj_num+"_prepr_data.pkl"
    with open(os.path.join(eeg_dir, subj_cat, subj_num, subj_data_fnm), 'rb') as file:
        subj_data = pickle.load(file)
    return subj_data

import pickle
import os
def load_stims_dict():
    stims_fnm = os.path.join("..",
                                "eeg_data",
                                'stims_dict.pkl')
    with open(stims_fnm, 'rb') as file:
        stims_dict = pickle.load(file)
    return stims_dict

from scipy import signal
from scipy.stats import pearsonr
import numpy as np
def match_waves(x, y, confidence_lims:list, fs:int):
    '''
    x: long waveform (1d array)
    y: shorter waveform (1d array)

    xcorr wrapper to find where in x y is likely to be
    assumes x is longer than y so that lagging y relative to x gives positive lag values 
    at the lag where the two signals overlap
    '''

    max_thresh, min_thresh = confidence_lims
    reduce_delta = 0.01
    confidence = 0.00
    thresh = max_thresh
    window_size = 2*y.size
    window_start = 0
    on_off_times = np.zeros(x.size, dtype=bool)
    # should find a clear peak if stim there
    print('checkpoint: before while loop')
    while confidence < thresh:
        if y.size > x[window_start:].size:
                #reached end of x, reduce threshold
                thresh -= reduce_delta 
                window_start = int(0) # restart from beginning with lower threshold
                # print(f"reducing threshold to {thresh}")
                
        if thresh <= min_thresh:
            # could not find stim, go on to next stim
            print(f"Could not find stim with min thresh. Skipping...")
            on_off_times = None
            break
            
        window_end = window_start + int(round(fs * window_size))
        if x[window_start:].size < y.size:
            raise NotImplementedError(f"Remaining recording envelope size is too small for stim size.")
            # on_off_times = None
            # break
        if window_end < x.size:
            r = signal.correlate(x[window_start:window_end], y, mode='valid')
            lags = signal.correlation_lags(x[window_start:window_end].size, y.size, mode='valid')
        else:
            r = signal.correlate(x[window_start:], y, mode='valid')
            lags = signal.correlation_lags(x[window_start:].size, y.size, mode='valid')

        
        
        sync_lag = lags[np.argmax(np.abs(r))]
        confidence = pearsonr(x[sync_lag:sync_lag+y.size], y).statistic
        if thresh == max_thresh:
            #only save on first run through recording
            confidence_val = confidence
        if sync_lag < 0 and confidence > thresh:
            raise NotImplementedError('Lags shouldnt be negative?')
        if confidence >= thresh:
            # print(f'Confidence exceeds threshold at lag={sync_lag}, recording timestamps.')
            #TODO: probably makes more sense to just keep the indices rather than full array
            on_off_times[sync_lag] += 1 #onset
            on_off_times[sync_lag+y.size] += 1 #offset
            # break

        # slide window until confidence threshold is reached
        window_start += int(round(fs * window_size/2))
    
    return on_off_times, confidence_val
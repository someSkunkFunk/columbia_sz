import numpy as np
def find_bad_electrodes(subj_data, criteria="4std"):
    # NOTE: this will implicitly drop missing trials as well
    if criteria == "4std":
        # reject electrodes whose std is over 4 times greater than median std
        # aggregate all stims into one array with dims [time x electrodes]
        R = np.concatenate([subj_data.dropna()['eeg'].loc[subj_data.dropna()['stim_nms']==nm].iloc[0] 
                            for nm in subj_data.dropna()["stim_nms"]])
        stds = np.std(R, axis=0)
        cutoff = 4*np.median(stds)
        outlier_idx = stds > cutoff

    return outlier_idx

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
import os
import pickle
import numpy as np
from utils import get_time_between_stims
def setup_xy(subj_data, stim_envs, subj_cat, subj_num, reduce_trials_by=None, outlier_idx=None):
    #TODO: get subj_cat and subj_num from subj_data (will have to add to subj data probably)
    stimulus = []
    response = []
    stim_nms = []
    prev_nm = None
    prev_block = None
     
    # need timestamps to reduce by pauses
    if reduce_trials_by == "pauses":
            # stitch trials together by block/pauses rather than stories;
            #  should give more than 18 trial
            fs_og = 2400 #NOTE: timestamps fs at og eeg fs but that info not in pickle file
            timestamps_fnm = os.path.join("..", 'eeg_data', "timestamps", subj_cat, subj_num, "timestamps.pkl")
            with open(timestamps_fnm, 'rb') as f:
                stim_start_end = pickle.load(f)
            times_between = get_time_between_stims(stim_start_end, fs_og)

    
    for stim_nm in subj_data.dropna()['stim_nms']:
        s = np.squeeze(stim_envs[stim_nm[:-4]])
        r = np.asarray(subj_data['eeg'].loc[subj_data['stim_nms']==stim_nm])[0]
        # filter out bad electrodes when given mask for outliers
        if outlier_idx is not None:
            r = r[:,~outlier_idx]
        # check that s, r lengths match, if not drop extra sample
        
        if abs(s.shape[0]- r.shape[0]) == 1:
            # sometimes they're off by one sample
            #NOTE: maybe check if the stimuli that are off by one sample correspond with those that have overlapping timestamps
            end = min([s.shape[0], r.shape[0]])
            # trim extra sample
            s, r = s[:end], r[:end,:]   
        if reduce_trials_by is not None:
            if reduce_trials_by == "stim_nm":
                if prev_nm is not None:
                    if stim_nm[:6] == prev_nm[:6]:
                        # concatenate to prev stim if in same block and story
                        stimulus[-1] = np.concatenate([stimulus[-1], s])
                        response[-1] = np.concatenate([response[-1], r])
                        stim_nms[-1].append(stim_nm)
                        prev_nm = stim_nm
                    else:
                        # not same story, make new list element
                        stimulus.append(s)
                        response.append(r)
                        stim_nms.append([stim_nm])
                        prev_nm = stim_nm
                else:
                    # first stim
                    stimulus.append(s)
                    response.append(r)
                    stim_nms.append([stim_nm])
                    prev_nm = stim_nm
            elif reduce_trials_by == "pauses":
                block = stim_nm[:3:2].capitalize()
                # stim_ps = times_between[block]
                if prev_nm is not None and prev_block == block:
                    # get number of samples between previous and current stim
                    pause_tm, _ = times_between[block][prev_nm+stim_nm]
                    if pause_tm <= 1:
                        # concatenate to prev stim if in same block and story
                        stimulus[-1] = np.concatenate([stimulus[-1], s])
                        response[-1] = np.concatenate([response[-1], r])
                        stim_nms[-1].append(stim_nm)
                        prev_nm = stim_nm
                    else:
                        # pause too long, make new list element
                        stimulus.append(s)
                        response.append(r)
                        stim_nms.append([stim_nm])
                        prev_nm = stim_nm
                else:
                    # first stim (or first block)
                    stimulus.append(s)
                    response.append(r)
                    stim_nms.append([stim_nm])
                    prev_nm = stim_nm
                prev_block = block
                    
                
        
        else:
            # Don't reduce trials at all
            stimulus.append(s)
            response.append(r)
            stim_nms.append([stim_nm])
    return stimulus, response, stim_nms
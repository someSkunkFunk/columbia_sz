import numpy as np
import scipy.io as spio
import os
from scipy import signal
from utils import get_lp_env, set_thresh, segment, get_stim_wav, match_waves
    
def get_timestamps(subj_eeg, eeg_dir, subj_num, choose_hc_sp, stims_dict, blocks):
    fs_audio = stims_dict['fs'][0] # 11025 foriginally #TODO: UNHARDCODE
    fs_eeg = 2400 #TODO: UNHARDCODE
    segment_blocks=False 
    which_corr = 'wavs' # 'wavs' or 'envs'
    # step size for xcorr window calculation
    confidence_lims = [0.90, 0.4] #max, min pearsonr correlation thresholds
    lpf_cut = 0.1 # lowpass cutoff for envelope-based segmetation

    # store indices for each block {"block_num":{"stim_nm": (start, end, rconfidence)}}
    stim_start_end = {}
    confidence_vals = []
    for block_num in blocks:
        stim_start_end[block_num] = {}
        # get stim order file
        stim_order_fnm = os.path.join(eeg_dir, choose_hc_sp, 
                                    subj_num, "original",
                                    "_".join([block_num, "stimorder.mat"])
                                    )
        block_stim_order = spio.loadmat(stim_order_fnm, squeeze_me=True)['StimOrder']
        # get recording envelope
        rec_wav = subj_eeg[block_num][:,-1]
        rec_env = np.abs(signal.hilbert(rec_wav))
        if segment_blocks:
            lp_rec_env = get_lp_env(rec_env, lpf_cut, fs_eeg)
            _, block_on_off = set_thresh(rec_wav, lp_rec_env, fs_eeg)
            if which_corr.lower() == 'envs':
                block_segments = segment(rec_env, block_on_off)
            elif which_corr.lower() =='wavs':
                block_segments = segment(rec_wav, block_on_off)
            else:
                raise NotImplementedError(f"which_corr = {which_corr} is not an option")
        else:
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
            # downsample current stim and get envelope  
            sos = signal.butter(8, fs_eeg/3, fs=fs_audio, output='sos')
            stim_wav = signal.sosfiltfilt(sos, stim_wav_og)
            # will "undersample" since sampling frequencies not perfect ratio
            stim_wav = signal.resample(stim_wav, int(np.floor(stim_dur*fs_eeg))) 
            stim_env = np.abs(signal.hilbert(stim_wav))
            if which_corr.lower() == 'wavs':
                y = stim_wav
            elif which_corr.lower() == 'envs':
                y = stim_env
            # prev_sync = int(0)
            if segment_blocks:
                for segment_ii, x in enumerate(block_segments):
                    if (x.size < stim_wav.size) and not (segment_ii+1) == len(block_segments):
                        # try next segment
                        continue
                    elif (x.size < stim_wav.size) and (segment_ii+1) == len(block_segments):
                        # stim missing and no confidence will be calculated
                        stim_start_end[block_num][stim_nm] = (None, None, None)

                    print("checkpoint: matching waves")
                    # NOTE: stim_on_off is relative to segment start, not full block recording
                    stim_on_off, confidence_val = match_waves(x, y, confidence_lims, fs_eeg)
                    print("checkpoint: waves matched")
                    if stim_on_off is not None:
                        # get start,end index for current stim relative to block start:
                        stim_start = np.where(stim_on_off)[0][0] + np.where(block_on_off)[0][0]
                        stim_end = np.where(stim_on_off)[0][1] + np.where(block_on_off)[0][0]
                        # save indices and confidence 
                        stim_start_end[block_num][stim_nm] = (stim_start, stim_end, confidence_val)
                        # don't look at recording up to this point again
                        block_segments[segment_ii] = x[np.where(stim_on_off)[0][0]:] 
                        # move onto next stim
                        break
                    else:
                        # missing stims
                        stim_start_end[block_num][stim_nm] = (None, None, confidence_val)
                        # move onto next stim
                        break
            else:
                if (x.size < stim_wav.size):
                    stim_start_end[block_num][stim_nm] = (None, None, None)
                    continue

                print("checkpoint: matching waves")
                stim_on_off, confidence_val = match_waves(x, y, confidence_lims, fs_eeg)
                print("checkpoint: waves matched")
                if stim_on_off is not None:

                    stim_start = np.where(stim_on_off)[0][0] + prev_end
                    stim_end = np.where(stim_on_off)[0][1] + prev_end
                    # save indices and confidence 
                    stim_start_end[block_num][stim_nm] = (stim_start, stim_end, confidence_val)
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
                    stim_start_end[block_num][stim_nm] = (None, None, confidence_val)
                    # move onto next stim
                    continue
    return stim_start_end
import os
import pickle
def load_stim_envs():
    stim_envs_pth=os.path.join("..","eeg_data","stim_envs.pkl")
    with open(stim_envs_pth, 'rb') as fl:
        stim_envs=pickle.load(fl)
    return stim_envs
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
def get_stim_envs(stims_dict, clean_or_noisy, fs_output, f_lp=49, filt_o=1):
    ''''
    stims_dict: classic stims dict with both dirty and clean stims from stim file
    clean_or_noisy: choose which stims to downsample and extract envelopes from "clean" or "noisy"
    fs_output: desired sampling frequency for resulting envelopes
    f_lp: low-pass filter cutoff, 30 Hz by default (must be below fs_output/2)
    returns: stim envelopes sampled at fs_output
    '''
    print(f"""Getting stim envelopes using low pass butter of order {filt_o} and cutoff at {f_lp} Hz""")
    if f_lp >= fs_output/2:
        raise NotImplementedError('low-pass is not below nyquist')
    fs_stim=stims_dict['fs'][0]
    sos = signal.butter(filt_o,f_lp,output='sos',fs=fs_stim)
    # get envelopes first, then resample
    stim_envs = {stim_nm: np.abs(signal.hilbert(stim_wav)) 
                 for stim_nm, stim_wav in zip(stims_dict['ID'], stims_dict["orig_"+clean_or_noisy])}

    #NOTE: getting envelopes first could be bad if resmple uses FFT method??
    ds_envs = {stim_nm: signal.resample(signal.sosfiltfilt(sos, stim_wav), int((stim_wav.size - 1)*fs_output/fs_stim)) 
                for stim_nm, stim_wav in stim_envs.items()}
    # rectify envelopes
    ds_envs = {stim_nm: np.maximum(s, 0) for stim_nm, s in ds_envs.items()}
    
    return ds_envs
import os
import pickle
import numpy as np
from utils import get_pause_times
def setup_xy(subj_data,stim_envs,subj_num,
              reduce_trials_by=None,outlier_idx=None,
              evnt=False,which_xcorr=None):
    '''
    NOTE SINCE SEPARATED AUDIO CHANNEL, 
    THIS FUNCTION WILL PROBABLY BREAK WHEN REDUCE_TRIALS_BY is NOT PAUSES
    BECAUSE IT"S THE ONLY CONDITION BLOCK I EDITED POST CHANGE
    subj_data: eeg 
    stim_envs: envelopes of individual waveforms presented during experiment
        should already be downsampled to eeg fs
    reduce_trials_by: method of selecting which stimuli/trials to concatenate into one 
            for computational purposes
    returns
    -
    stimulus: concatenated stim envelopes
    response:
    stim_nms:
    '''
    #TODO: get subj_cat and subj_num from subj_data (will have to add to subj data probably)
    stimulus = []
    response = []
    stim_nms = []
    audio_recorded=[]
    prev_nm = None
    prev_block = None
    if evnt:
        # subj_data has different structure -> just a python dictionary with segments already
        for ii, (_,seg_data) in enumerate(subj_data.items()):
            # assumes overly-disparate stim/response pairs (in terms of duration)
            # are already taken out of subj_data, so can just pad pairs that are uneven here
            audio_recorded.append(seg_data[1])
            stim_nms.append([nm.strip('.wav') for nm in seg_data[0]])
            s=np.concatenate([stim_envs[nm] for nm in stim_nms[ii]])
            r=seg_data[-1]
            size_diff=s.shape[0]-r.shape[0]
            if size_diff==0:
                stimulus.append(s)
                response.append(r)
            elif size_diff>0:
                #stim is longer, pad response at the end
                pd_wdth=((0,size_diff),)
                r=np.pad(r,pd_wdth)
                response.append(r)
                stimulus.append(s)
            elif size_diff<0:
                # response is longer, pad stim at end
                pd_wdth=((0,abs(size_diff)),)
                s=np.pad(s,pd_wdth)
                stimulus.append(s)
                response.append(r)
            assert s.shape[0]-r.shape[0] == 0 , "padding should have made these equal by now!"



        pass
    elif not evnt:    
        for stim_nm in subj_data.dropna()['stim_nms']:
            s=np.squeeze(stim_envs[stim_nm[:-4]])
            r=np.asarray(subj_data['eeg'].loc[subj_data['stim_nms']==stim_nm])[0]
            audio_bit=np.asarray(subj_data['eeg_audio'].loc[subj_data['stim_nms']==stim_nm])[0]
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
                    # stitch trials together by block/pauses rather than stories;
                    #  should give more than 18 trial
                    fs_timestamps = 2400 #NOTE: timestamps fs at og eeg fs but that info not in pickle file
                    if evnt:
                        which_timestamps="evnt"
                    else:
                        which_timestamps="mine"
                    
                    pause_times = get_pause_times(subj_num,which_timestamps,fs_timestamps,which_xcorr=which_xcorr)
                    block = stim_nm[:3:2].capitalize()
                    # stim_ps = times_between[block]
                    if prev_nm is not None and prev_block == block:
                        # get number of samples between previous and current stim
                        pause_tm, _ = pause_times[block][prev_nm+stim_nm]
                        if pause_tm <= 1:
                            # concatenate to prev stim if in same block and story
                            stimulus[-1]=np.concatenate([stimulus[-1], s])
                            response[-1]=np.concatenate([response[-1], r])
                            stim_nms[-1].append(stim_nm)
                            audio_recorded[-1]=np.concatenate([audio_recorded[-1],
                                                            audio_bit])
                            prev_nm = stim_nm
                        else:
                            # pause too long, make new list element
                            stimulus.append(s)
                            response.append(r)
                            stim_nms.append([stim_nm])
                            audio_recorded.append(audio_bit)
                            prev_nm = stim_nm
                    else:
                        # first stim (or first block)
                        stimulus.append(s)
                        response.append(r)
                        stim_nms.append([stim_nm])
                        audio_recorded.append(audio_bit)
                        prev_nm = stim_nm
                    prev_block = block
                        
                    
            
            else:
                # Don't reduce trials at all
                stimulus.append(s)
                response.append(r)
                stim_nms.append([stim_nm])
                
    return stimulus, response, stim_nms, audio_recorded
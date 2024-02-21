def check_timestamps(timestamps):
    '''
    helper to make sure timestamps are non-overlapping and strictly ascending
    '''
    for block in timestamps:
        print(f"checking block {block}:")
        nms=[]
        starts=[]
        ends=[]
        for nm,(start,end,_) in timestamps[block].items():
            if all([start,end]):
                nms.append(nm)
                starts.append(start)
                ends.append(end)
        #convert to arrays for mathing
        starts=np.array(starts)
        ends=np.array(ends)
        #check non-overlapping:
        if np.all((starts[1:]-ends[:-1])>=0) and\
            starts[0]==np.min([starts,ends]) and\
            ends[-1]==np.max([starts,ends]):
            print("no indices overlap")
        else:
            n_overalapping=np.sum((starts[1:]-ends[:-1])<0)
            #note: ignoring start and end; this should trivially not overlap anyway
            print(f"{n_overalapping} indices overlap.")
    
        if np.all((ends>starts)):
            print("all end indices larger than start.")
        else:
            print(f"{np.sum(ends<=starts)} stims have smaller end indices than start.")
        if np.all(np.diff(starts)>=0) and np.all(np.diff(ends)>=0):
            print("all starts, ends ascending.")
        else:
            print(f"{np.sum(np.diff(starts)<0)} starts not ascending, \
                  {np.sum(np.diff(ends)<0)} ends not ascending.")



import os
def get_all_subj_nums(single_cat=None):
    '''
    single_cat=None (get all, default), "hc", "sp"
    helper to get all subj nums from raw directory folder structure;
    returns: list of all subj nums, 
    '''
    eeg_dir=os.path.join('..',"eeg_data","raw")
    hc_pth=os.path.join(eeg_dir,"hc")
    sp_pth=os.path.join(eeg_dir,"sp")
    if single_cat is None:
        all_subjs=os.listdir(hc_pth)+os.listdir(sp_pth)
    elif single_cat=='hc':
        all_subjs=os.listdir(hc_pth)
    elif single_cat=='sp':
        all_subjs=os.listdir(sp_pth)
    else:
        raise NotImplementedError(f'{single_cat} is not an option for <single_cat>')
    return all_subjs

import numpy as np
import pandas as pd
def align_responses(subj_eeg:dict, timestamps_tup:tuple, audio_chn, stims_dict:dict):
    '''
    subj_eeg: {'block_num': [time x channels] - eeg numpy array}
    timestamps: tuple of {'block_num': {'stim_nm': (onset_time, offset_time)} } 
                dicts for downsampled and og timestamps
    stims_dict: master stim directory with original stim wav files (noisy and clean)
    returns
    pandas dataframe with aligned eeg data
        columns: 
                stim_nms
                eeg: numpy matrix, contains np.nans for stimuli with no start pt
                eeg_audio
                confidence
    '''
    #TODO: add optional padding around on/of time edges 
    #NOTE: stims_dict not being used by/needed function 
    # but kinda useful during debugging
    
    ts_og,ts_ds=timestamps_tup
    S_names=[]
    confidence_vals=[]
    R=[]
    R_audio=[]
    for block in subj_eeg:
        eeg,audio_rec=subj_eeg[block]
        for stim_nm, (start, end, confidence) in ts_ds[block].items():
            S_names.append(stim_nm)
            confidence_vals.append(confidence)
            if all([start,end]):
                R.append(eeg[start:end,:])
                # get original timestamps since audio recording is not downsapled
                start_og,end_og,_=ts_og[block][stim_nm]
                R_audio.append(audio_rec[start_og:end_og])
            else:
                R.append(np.nan)
                R_audio.append(np.nan)
            if confidence is None:
                #NOTE: this should actually never happen because we are always 
                # storing SOME confidence value even when low
                # added to find out why subj_data after preprocessing and segmenting has some None values
                print(f"Confidence is None for {stim_nm} in utils.align_responses()!")
    S_names = np.array(S_names)
    # timestamps = np.array(timestamps)
    confidence_vals = np.array(confidence_vals)

    
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
    ASSUMES DATA FOR ALL 6 BLOCKS IS PRESENT IN RAW FOLDER
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
    #NOTE LISTDIR DOESN'T GIVE THEM IN ORDER, sorting assumes block data files names consistent acrosss subjects
    eeg_fnms.sort()
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

def get_stim_wav(stims_dict, stim_nm:str, noisy_or_clean='clean'):
    if "noisy_or_clean" in os.environ:
        noisy_or_clean=os.environ['noisy_or_clean']
        # print(f'yay this worked, noisy_or_clean is {noisy_or_clean}')
    if stim_nm.lower().endswith('.wav'):
        # remove '.wav' from name to match stim mat file
        stim_indx = stims_dict['ID'] == stim_nm[:-4]
    else:
        # assume just string of name
        stim_indx = stims_dict['ID'] == stim_nm
    
    return stims_dict[f'orig_{noisy_or_clean}'][stim_indx][0]

import scipy.io as spio
# from .mat2dict import mat2dict
def get_stims_dict(stim_fl_pth=None):
    '''
    transform mat file to dictionary for 
    '''
    # NOTE: determined that both stim files have same orig_noisy stim wavs
    if stim_fl_pth is None:
        stim_fl_pth=os.path.join("..","eeg_data","stim_info.mat")
    all_stims_mat=spio.loadmat(stim_fl_pth, squeeze_me=True)

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
 
def get_pause_times(subj_num,which_timestamps,fs,which_xcorr=None):
    '''
    note that when using evnt timestamps, only includes those 
    whose evnt structure confidence value is over 0.4, 
    probably smarter to put this condition either in preprocessing 
    (as done with our own timestamps) or in trf_helpers
    returns
    fs is sampling frequency of timestamps
    pause_times
    NOTE im pretty sure sample calculated in time are incorrect (off by 1)
    but setupxy is using the samples number anyway
    '''
    subj_cat=get_subj_cat(subj_num)
    if which_timestamps=="mine":
        ts_pth=os.path.join("..","eeg_data","timestamps",subj_cat,subj_num,
                            f"{which_xcorr}_timestamps.pkl")
        with open(ts_pth, 'rb') as f:
            timestamps=pickle.load(f)
        # my timestamps are organized hierarchically by block, evnt are not
        pause_times = {}
        for block in timestamps.keys():
            pause_times[block] = {} 
            #NOTE: I believe start/end are sample indices but should double check
            prev_end = None
            prev_stim_nm = None
            for stim_nm, (start, end, _) in timestamps[block].items():
                if start is not None:
                    if prev_end is None:
                        # first stim
                        prev_end = end
                        prev_stim_nm = stim_nm
                    else:
                        # record transition pause time in samples and time
                        trans_nm = prev_stim_nm+stim_nm
                        pause_times[block][trans_nm]=(int(start-prev_end), 
                                                      (start-prev_end)/fs)
                        prev_end = end
                        prev_stim_nm = stim_nm
                else:
                    # skip missing stims
                    continue 
    elif which_timestamps=="evnt":
        ts_pth=os.path.join("..","eeg_data","timestamps",
                            f"evnt_{subj_num}.mat")
        evnt_mat=spio.loadmat(ts_pth)
        # returns dict for some reason, which mat2dict doesnt like
        evnt=evnt_mat['evnt']
        timestamps=mat2dict(evnt)
        del evnt, evnt_mat
        pause_times={}
        shift=12000 #empirical shift between their timestamps and mine in samples 
        confidence_thresh=0.4
        #NOTE: i don't like having to reload stims_dict for this part since it requires a lot of memory
        # but until we save stim durations somewhere independent of wav files this is necessary to get
        # end index for each stimulus (we can't trust the start/stop times in evnt yet 
        # since unsure about consistency of shift)
        stims_dict=get_stims_dict()
        # fs_audio=stims_dict['fs'][0]
        fs_audio=16000
        for stim_nm in timestamps['name'][0]:
            # prev_nm=None
            # prev_end=None
            stim_ii=timestamps['name'][0]==stim_nm
            stim_nm_str=stim_nm[0] # get str out of arr
            curr_block=stim_nm_str[:3].replace("0","").capitalize()
            if curr_block not in pause_times.keys():
                # new block starts here
                pause_times[curr_block]={}
                prev_start=None
                prev_end=None
            # check confidence threshold is met

            if timestamps['confidence'][0,stim_ii][0][0][0]>confidence_thresh:
                curr_start=timestamps['syncPosition'][0,stim_ii][0][0][0]+shift
                stim_wav=get_stim_wav(stims_dict,stim_nm_str,'clean')
                stim_dur=(stim_wav.size-1)/fs_audio
                del stim_wav
                stim_len=int(stim_dur*fs+1)
                curr_end=curr_start+stim_len
                if prev_end is None:
                    #first stim in block
                    prev_end=curr_end
                    prev_nm=stim_nm_str
                else:
                    # record transition pause time in samples and time
                    trans_nm=prev_nm+stim_nm_str
                    pause_times[curr_block][trans_nm]=(int(curr_start-prev_end), 
                                                       (curr_start-prev_end)/fs)
                    prev_end=curr_end
                    prev_nm=stim_nm_str
            else:
                continue #skip missing stims

    return pause_times

import numpy as np
import scipy.io as spio
import os
from scipy import signal
# NOTE: don't need imports from utils anymore since on same module?
# from utils import get_lp_env, set_thresh, segment, get_stim_wav, match_waves    
def get_timestamps(subj_eeg,eeg_dir,subj_num,subj_cat,stims_dict,blocks,
                   which_xcorr='wavs',confidence_lims=[0.80, 0.4]):
    '''
    uses xcorr to find where recorded audio best matches 
    stimuli waveforms (which_corr='wavs', deault) or envelopes (which_corr='envs)
    If sync point gives pearsonr confidence above minimum threshold, indices for onset/offset 
    and confidence are stored in timestamps directory (keys are blocks) as thruples
    if not above threshold, start and end are None but still returns confidence for max sync point found...
    '''
    fs_audio=stims_dict['fs'][0] # 11025 foriginally #TODO: UNHARDCODE
    # fs_audio=16000
    fs_eeg=2400 #TODO: UNHARDCODE
    
    # store indices for each block {"block_num":{"stim_nm": (start, end, rconfidence)}}
    timestamps = {}
    for block_num in blocks:
        print(f'timestamping for block: {block_num}')
        timestamps[block_num] = {}
        # get stim order file
        stim_order_fnm = os.path.join(eeg_dir, subj_cat, 
                                    subj_num, "original",
                                    "_".join([block_num, "stimorder.mat"])
                                    )
        block_stim_order = spio.loadmat(stim_order_fnm, squeeze_me=True)['StimOrder']
        # get experiment audio recording envelope
        rec_wav = subj_eeg[block_num][:,-1]
        if which_xcorr.lower() == 'envs':
            rec_env = np.abs(signal.hilbert(rec_wav))
            x = rec_env
        elif which_xcorr.lower() =='wavs':
            x = rec_wav
        else:
            raise NotImplementedError(f"which_corr = {which_xcorr} is not an option")
        prev_end=0 #TODO: verify this doesn't introduce new error (check w finished subject?)
        for stim_ii, stim_nm in enumerate(block_stim_order):
            print(f"finding {stim_nm} ({stim_ii+1} of {block_stim_order.size})")
            # grab stim wav
            #NOTE: not sure if get_stim_wav will overwrite based on os environment vars
            stim_wav_og=get_stim_wav(stims_dict, stim_nm)
            stim_dur = (stim_wav_og.size - 1)/fs_audio
            #TODO: what if window only gets part of stim? 
            # apply antialiasing filter to stim wav and get envelope  
            sos = signal.butter(3, fs_eeg/3, fs=fs_audio, output='sos')
            stim_wav = signal.sosfiltfilt(sos, stim_wav_og)
            stim_env = np.abs(signal.hilbert(stim_wav))
            # downsample envelope or wav to eeg fs
            if which_xcorr.lower() == 'wavs':
                # will "undersample" since sampling frequencies not perfect ratio
                y = signal.resample(stim_wav, int(np.floor(stim_dur*fs_eeg)+1)) 
            elif which_xcorr.lower() == 'envs':
                # will "undersample" since sampling frequencies not perfect ratio
                y = signal.resample(stim_env, int(np.floor(stim_dur*fs_eeg)+1)) 
                
            if (x.size < y.size):
                timestamps[block_num][stim_nm] = (None, None, None)
                continue

            print("matching waves")
            if which_xcorr=='wavs':
                # concerned that maybe standardizing makes problem harder when using envelopes
                stim_on_off,confidence_val,over_thresh=match_waves(x,y,confidence_lims,fs_eeg)
            elif which_xcorr=='envs':
                stim_on_off,confidence_val,over_thresh=match_waves(x,y,confidence_lims,fs_eeg,
                                                                   standardize=False)

            print(f"confidence_val: {confidence_val}, above threshold: {over_thresh}")
            #NOTE: code below used to depend on stim_on_off being none, but now checks over_thresh to decide 
            # if timestamps should be recorded
            if over_thresh:

                curr_start = np.where(stim_on_off)[0][0] + prev_end
                curr_end = np.where(stim_on_off)[0][1] + prev_end
                print(f"number of samples between detected endpoints: {curr_end-curr_start}")
                # save indices and confidence 
                timestamps[block_num][stim_nm] = (curr_start, curr_end, confidence_val)
                # don't look at recording up to this point again
                if which_xcorr=="wavs":
                    x=rec_wav[curr_end:]
                elif which_xcorr=="envs":
                    x=rec_env[curr_end:]
                print(f"size of x after new startpoint:{x.size}")
                # mark end time of last stim found
                # NOTE: not sure if this will work if first stim in block can't be found
                prev_end = curr_end
                if stim_ii==100:
                    pass
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
def load_preprocessed(subj_num,eeg_dir=None,evnt=False,which_xcorr=None):
    '''
    helper to load segmented and preprocessed
    subject data as pandas dataframe (from pkl fl)
    '''
    subj_cat = get_subj_cat(subj_num)
    if eeg_dir is None:
        if evnt==False:
            eeg_dir=os.path.join(os.getcwd(), '..', "eeg_data", "preprocessed_xcorr")
        elif evnt:
            eeg_dir=os.path.join(os.getcwd(), '..', "eeg_data", "preprocessed_evnt")
        else:
            raise NotImplementedError(f"evnt: {evnt} error")
    if which_xcorr is None and evnt:
        subj_data_fnm = "aligned_resp.pkl"
    else:
        # only xcorr-aligned data has which_xcorr prefix 
        subj_data_fnm = f"{which_xcorr}_aligned_resp.pkl"
    with open(os.path.join(eeg_dir, subj_cat, subj_num, subj_data_fnm), 'rb') as file:
        subj_data = pickle.load(file)
    return subj_data


# discontinuing use of this convenience function because it is getting inconvenient to have this and get_stims_dict
# plus i dont have the pkl fl saved kinda superflous
# import pickle
# import os
# def load_stims_dict():
#     ##TODO: needa re-save stim_info.mat as dict for this to work
#     stims_fnm = os.path.join("..",
#                                 "eeg_data",
#                                 'stims_dict.pkl')
#     with open(stims_fnm, 'rb') as file:
#         stims_dict = pickle.load(file)
#     return stims_dict

from scipy import signal
from scipy.stats import pearsonr
import numpy as np
def match_waves(x, y, confidence_lims:list, fs:int, standardize=True):
    '''
    x: long waveform (1d array)
    y: shorter waveform (1d array)

    xcorr wrapper to find where in x y is likely to be
    assumes x is longer than y so that lagging y relative to x gives positive lag values 
    at the lag where the two signals overlap
    '''
    #TODO: only one threshold is probably necessary
    max_thresh, min_thresh = confidence_lims
    current_confidence = 0.00
    max_confidence=0.00
    thresh = max_thresh
    on_off_times=np.zeros(x.size, dtype=bool)
    exceed_thresh=False
    if standardize:
        x=(x-np.mean(x))/np.std(x)
        y=(y-np.mean(y))/np.std(y)
    # should find a clear peak if stim there
    for thresh in confidence_lims:
            
        if x.size<y.size:
            raise NotImplementedError(f"Remaining recording size is too small for stim size.")

        r=signal.correlate(x,y,mode='valid')
        lags=signal.correlation_lags(x.size,y.size,mode='valid')

        
        #NOTE: not off my one.... i think
        sync_lag=lags[np.argmax(np.abs(r))]
        # calculate pearson corr between segments
        x_segment=x[sync_lag:sync_lag+y.size]
        current_confidence = abs(pearsonr(x_segment, y).statistic)
        if thresh==max_thresh and current_confidence>=max_confidence:
            #only save on first run through recording
            #update return value and update max to beat
            max_confidence=current_confidence

            # reset timestamps to zero, update with max confidence lags
            try:
                # NOTE: if onset is still within range on_off_times, 
                # on_off_times may still contain an onset timestamp
                on_off_times[:]=0 
                on_off_times[sync_lag]+=1 #onset
                on_off_times[sync_lag+y.size]+=1 #offset
            except IndexError:
                print(f'onset,offset lags of: {sync_lag,sync_lag+y.size} are out of range for x.size: {x.size}')
        elif thresh < max_thresh and current_confidence==max_confidence:
            #not functional except to observe what's happening with xcorr around maximum lag
            pass
        if sync_lag<0 and current_confidence>thresh:
            raise NotImplementedError('Lags shouldnt be negative?')
        if current_confidence>=thresh:
            # print(f'Confidence exceeds threshold at lag={sync_lag}, recording timestamps.')
            exceed_thresh=True
        if thresh==min_thresh and not exceed_thresh:
            # could not find stim, go on to next stim
            print(f"Could not find stim with min thresh. Skipping...")
        


    
    return on_off_times, max_confidence, exceed_thresh
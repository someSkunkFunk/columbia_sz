#%%
from scipy import signal
import numpy as np
def get_segments(envelope:np.ndarray,fs,params=None):
    '''
    envelope: broadband envelope, assumed to be a 1D array
    function that uses smoothed envelope thresholding to get indices for non-silent segments
    in which match waves will look for stimuli
    params: {
    'filt_ord': sharpness of filter applied on smoothing envelope
    'filt_freqs': lowpass filter frequency (or list bandpass limits if bandpass desired)
    'min_sil': minimum silent period duration in seconds; 
    }

    returns
    -
    smooth_envelope: smoothed envelope normalized between 0 and 1
    segments: [n_segments x 2] array with first col containing indices of 
                                segment start and second col the end
    '''
    smooth_envelope=envelope.copy() #NOTE: useful during debugging but won't need once finished
    assert smooth_envelope.ndim==1, "this function assumes 1D array!"
    if params is None:
        params={
            'filt_ord':16,
            'filt_freqs':0.1,
            'min_sil': 3,
            'seg_padding': 0.5
        }
    if isinstance(params['filt_freqs'], list) and len(params['filt_freqs'])==2:
        _filt_type='bandpass'
    elif isinstance(params['filt_freqs'],float):
        _filt_type='low'
    else:
        raise NotImplementedError(f"params['filt_freqs'] should be float or int-> {type(params['filt_freqs'])}")
    # rectify since we looking for overall magnitude changes
    smooth_envelope[smooth_envelope<0]*=-1.0
    # smooth the envelope
    sos=signal.butter(params['filt_ord'],params['filt_freqs'],btype=_filt_type,output='sos',fs=fs)
    smooth_envelope=signal.sosfiltfilt(sos,smooth_envelope)
    # normalize between zero and 1
    smooth_envelope=(smooth_envelope-smooth_envelope.min())/(smooth_envelope.max()-smooth_envelope.min())
    
    assert np.all(smooth_envelope<=1) and np.all(smooth_envelope>=0), "envelope range should be between 0 and 1 here!"
    # get indices where smoothed envelope crosses half-median, should be 1 where envelope goes above .5 
    # the range and -1 where it goes back below
    loud_bits=(smooth_envelope>0.5*np.median(smooth_envelope)).astype(int)
    crossings=np.diff(loud_bits,prepend=1) #note: prepend w 1 to avoid introducing artifactual onset at beginning
    # remove spurious crossings at the beginning/end
    if np.argwhere(crossings==1)[0] > np.argwhere(crossings==-1)[0]:
        crossings[np.argwhere(crossings==-1)[0]]=0
    if np.argwhere(crossings==1)[-1] > np.argwhere(crossings==-1)[-1]:
        crossings[np.argwhere(crossings==1)[-1]]=0

    # separate onsets from offsets, then pad 
    # past start/end of recording array size
    # pad onsets by removing number of samples equal to seg_padding
    n_shift=int(params['seg_padding']*fs)
    if np.any(crossings[:n_shift]==1):
        raise NotImplementedError("shifting first onset too far!")
    if np.any(crossings[-n_shift:]==-1):
        raise NotImplementedError("shifting last offset too far!")
    onsets=np.concatenate((crossings[n_shift:]==1, np.zeros(n_shift)))
    offsets=np.concatenate((np.zeros(n_shift), crossings[:-n_shift]==-1))
    pause_durs=(np.argwhere(onsets)[1:]-np.argwhere(offsets)[:-1])/fs #in seconds
    assert np.all(pause_durs>0), "all pauses durations should be positive (before removing short ones)!"
    # remove excessively short pauses
    if np.any(pause_durs<params['min_sil']):
        rmv_indx=pause_durs<params['min_sil']
        onsets[np.argwhere(onsets)[1:][rmv_indx]]=0
        offsets[np.argwhere(offsets)[:-1][rmv_indx]]=0
    pause_durs=(np.argwhere(onsets)[1:]-np.argwhere(offsets)[:-1])/fs
    assert np.all(pause_durs>0), "all pauses durations should be positive (after removing short ones)!"
    segments=np.hstack([np.argwhere(onsets),np.argwhere(offsets)])
    #TODO: check that segments is n x 2 array
    return segments, smooth_envelope


#%%

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
                   thresh_params,which_xcorr='wavs'):
    '''
    uses xcorr to find where recorded audio best matches 
    stimuli waveforms (which_corr='wavs', deault) or envelopes (which_corr='envs)
    If sync point gives pearsonr confidence above minimum threshold, indices for onset/offset 
    and confidence are stored in timestamps directory (keys are blocks) as thruples
    if not above threshold, start and end are None but still returns confidence for max sync point found...
    '''
    plot_failures=False
    fs_audio=stims_dict['fs'][0] # 11025 foriginally #TODO: UNHARDCODE
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
        # get experiment audio recording envelope, after de-trending via highpass above 1 Hz
        rec_wav = subj_eeg[block_num][:,-1]
        filt_wav=sos=signal.butter(3,1.0,fs=fs_eeg,btype='high',output='sos')
        filt_wav=signal.sosfiltfilt(sos,rec_wav,)
        rec_env = np.abs(signal.hilbert(filt_wav))
        
        if which_xcorr.lower() == 'envs':
            x=rec_env.copy()
            standardize=False
        elif which_xcorr.lower() =='wavs':
            x=rec_wav.copy()
            standardize=False #NOTE: setting to false because now I'm zeoring out silent periods to get rid of spurious correlations and standardize is going to move the zero period I belive
        else:
            raise NotImplementedError(f"which_corr = {which_xcorr} is not an option")
        
        # get segments where sound happened:
        print(f"splitting sound recording into segments")
        segments, smooth_envelope=get_segments(rec_env,fs_eeg)
        n_segments=segments.shape[0]
        print(f"{n_segments} segments found.")
        #### DEBUG PLOTTING###
        subj_debug_dir=os.path.join("..","figures","debug",subj_num)
        if not os.path.isdir(subj_debug_dir):
            os.makedirs(subj_debug_dir,exist_ok=True)
        import matplotlib.pyplot as plt
        t=np.arange(smooth_envelope.size)/fs_eeg
        on_times=t[segments[:,0]]
        off_times=t[segments[:,1]]
        plt.plot(t,filt_wav/np.max(np.abs(filt_wav)),label='highpassed recording')
        plt.plot(t,smooth_envelope,label='smooth_env')
        plt.stem(on_times,np.ones(on_times.shape),label='onsets',linefmt='green')
        plt.stem(off_times,np.ones(off_times.shape),label='offsets',linefmt='red')
        plt.xlabel('seconds')
        plt.legend(loc='lower left')
        plt.title(f"{subj_num} {block_num}")
        plt.tight_layout()
        fig_fnm=f"{subj_num}_{block_num}_segments.png"
        fig_pth=os.path.join(subj_debug_dir,fig_fnm)
        plt.savefig(fig_pth)
        

        for ii, (on,off) in enumerate(segments):
            plt.figure()
            recording_bit=filt_wav[on:off]
            recording_bit/=np.abs(recording_bit).max()
            plt.plot(t[on:off],recording_bit,label='highpassed recording')
            plt.title(f"{subj_num} {block_num} segment {ii+1}")
            plt.plot(t[on:off],smooth_envelope[on:off],label='smooth_env')
            plt.legend()
            plt.xlabel('seconds')
            plt.tight_layout()
            fig_fnm=f"{subj_num}_{block_num}_segment{ii+1}.png"
            fig_pth=os.path.join(subj_debug_dir,fig_fnm)
            plt.savefig(fig_pth)
        #### END DEBUG PLOTTING###TODO: MAKE INTO A FUNCTION>>
        current_segment=0
        last_found_lag=0
        # TODO: iterate thru each segment and do match waves
        for stim_ii, stim_nm in enumerate(block_stim_order):
            if current_segment == n_segments-1:
                print("end of final segment reached, breaking loop, going to next block")
                break
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
                
            # if (x.size < y.size):
            #     timestamps[block_num][stim_nm] = (None, None)
            #     continue
            segment_start=segments[current_segment][0]
            segment_end=segments[current_segment][1]
            startpoint=segment_start+last_found_lag
            print(f"matching waves, STARTPOINT:{startpoint}")
            sync_lag,over_thresh=match_waves(x[startpoint:segment_end+1],y,fs_eeg,
                                             thresh_params,standardize=standardize)
 
            print(f"waves_matched, above threshold: {over_thresh}")
            if over_thresh is None and plot_failures:
                print('renaming failure case figure file...')
                # rename failure case figure to something more informative
                fig_pth=os.path.join("..","figures","debug","failure_case.png")
                #NOTE: os rename might not work as expected here
                new_pth=os.path.join("..","figures","debug",f"{subj_num}_{stim_nm[:-4]}_{stim_ii:03d}.png")
                os.rename(fig_pth,new_pth)
                print(f"{fig_pth} -> {new_pth} complete.")
                # record timestamps as missing
                timestamps[block_num][stim_nm] = (None, None)
            elif over_thresh is None and not plot_failures:
                timestamps[block_num][stim_nm] = (None, None)

            #NOTE: code below used to depend on stim_on_off being none, but now checks over_thresh to decide 
            # if timestamps should be recorded
            if over_thresh:
                curr_start=segment_start+last_found_lag+sync_lag
                curr_end=curr_start+y.size
                # save indices 
                timestamps[block_num][stim_nm]=(curr_start,curr_end)
                # TODO: check that last_found_lag actually refers to the sample index where most recently found stim was
                last_found_lag=curr_end-segment_start #need to subtract segment_start because we want index relative to segment
                if last_found_lag >= (segment_end-segment_start):
                    # reached end of segment
                    current_segment+=1
                print(f"found {stim_nm} with size {y.size} in segment {current_segment+1}; ends at segment lag: {last_found_lag}.")
            else:
                # missing stims
                timestamps[block_num][stim_nm] = (None, None)
                print(f"failed to find {stim_nm}, moving on to next stim.")
                # move onto next stim
                
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
def match_waves(x, y, fs:int, thresh_params:tuple, standardize=True):
    '''
    x: long waveform (1d array)
    y: shorter waveform (1d array)

    xcorr wrapper to find where in x y is likely to be
    assumes x is longer than y so that lagging y relative to x gives positive lag values 
    at the lag where the two signals overlap
    '''
    if thresh_params[0].lower()=='xcorr_peak':
        cutoff_ratio=thresh_params[1]
        # cutoff adapts based on overall xcorr stats
    elif thresh_params[0].lower()=='pearsonr':
        # use fixed pearsonr threshold
        cutoff_confidence=thresh_params[1]

        
    exceed_thresh=False
    stim_lag=None
    if standardize:
        x=(x-np.mean(x))/np.std(x)
        y=(y-np.mean(y))/np.std(y)
    # should find a clear xcorr peak if stim there
    
        
    if x.size<y.size:
        raise NotImplementedError(f"Remaining recording size is too small for stim size.")

    r=signal.correlate(x,y,mode='full')
    lags=signal.correlation_lags(x.size,y.size,mode='full')
    #ignore negative lags
    r=r[(y.size-1):]
    lags=lags[(y.size-1):]

    
    
    max_xcorr=np.max(np.abs(r))
    sync_lag=lags[np.argmax(np.abs(r))]
    # calculate pearson corr between segments
    x_segment=x[sync_lag:sync_lag+y.size]
    if thresh_params[0]=='xcorr_peak':
        # use xcorr peak as confidence indicator
        xcorr_cutoff=cutoff_ratio*np.std(r)
        if x_segment.size!=y.size:
            # something went wrong and we want to plot it to figure out why
            import matplotlib.pyplot as plt
            exceed_thresh=None #TODO: get more information out of get_timestamps when this is None
            fig,ax=plt.subplots(3,1)
            ax[0].plot(lags,r,label=f"sync_lag={sync_lag}")
            ax[0].axhline(xcorr_cutoff,color="red",label=f"cutoff ratio:{cutoff_ratio}")
            ax[0].axhline(-xcorr_cutoff,color="red")
            ax[0].set_title('xcorr')
            ax[0].legend()
            ax[0].set_xlabel('lags, samples')

            tx=np.arange(x.size)/fs
            ax[1].plot(tx,x,)
            ax[1].set_xlabel('time, s')
            ax[1].set_title('remaining sound recording')
            

            ty=np.arange(y.size)/fs
            tx_s=np.arange(x_segment.size)/fs
            ax[2].plot(ty,y/np.max(np.abs(y)),label="stim")
            ax[2].plot(tx_s,x_segment/np.max(np.abs(x))+1,label="x_segment")
            ax[2].plot(tx[:y.size],x[:y.size]/np.max(np.abs(x))+2,label="x[:y.size]")
            ax[2].legend()
            ax[2].set_xlabel('time, s')

            plt.tight_layout()
            save_pth=os.path.join("..","figures","debug","failure_case.png")
            plt.savefig(save_pth,format='png')
            print("failure case figure saved")
        if max_xcorr>xcorr_cutoff:
            stim_lag=sync_lag
            exceed_thresh=True
        elif max_xcorr<=xcorr_cutoff and sync_lag==0:
            stim_lag=sync_lag
            exceed_thresh=True
        else:
            print(f"Could not find stim with min thresh. Skipping...")
    elif thresh_params[0]=='pearsonr':
        if x_segment.size!=y.size:
            print("remaining recording length too short...")
            #plot the fucking bullshit
            # import matplotlib.pyplot as plt
            # exceed_thresh=None #TODO: get more information out of get_timestamps when this is None
            # fig,ax=plt.subplots(3,1)
            # ax[0].plot(lags,r,label=f"sync_lag={sync_lag}")
            # ax[0].set_title('xcorr')
            # ax[0].legend()
            # ax[0].set_xlabel('lags, samples')

            # tx=np.arange(x.size)/fs
            # ax[1].plot(tx,x,)
            # ax[1].set_xlabel('time, s')
            # ax[1].set_title('remaining sound recording')
            

            # ty=np.arange(y.size)/fs
            # tx_s=np.arange(x_segment.size)/fs
            # ax[2].plot(ty,y/np.max(np.abs(y)),label="stim")
            # ax[2].plot(tx_s,x_segment/np.max(np.abs(x))+1,label="x_segment")
            # ax[2].plot(tx[:y.size],x[:y.size]/np.max(np.abs(x))+2,label="x[:y.size]")
            # ax[2].legend()
            # ax[2].set_xlabel('time, s')

            # plt.tight_layout()
            # save_pth=os.path.join("..","figures","debug","failure_case.png")
            # plt.savefig(save_pth,format='png')
            # print("failure case figure saved")
            
        else:
            # matched appropriately sized segment, proceed with caution
            current_confidence=np.abs(pearsonr(x_segment, y).statistic)
            if current_confidence>cutoff_confidence:
                stim_lag=sync_lag
                exceed_thresh=True
            else:
                pass
                # plot sub-threshold segments to see
                # print(f"Could not find stin with min thresh... plotting sync lag segment") 
                # import matplotlib.pyplot as plt
                # fig,ax=plt.subplots()
                # ax.plot()
    return stim_lag, exceed_thresh
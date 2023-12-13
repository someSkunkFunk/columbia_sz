def get_subj_cats(eeg_dir):
    """
    create dictionary for looking up subject number based on healthy control (hc) or 
    schizophrenia patient (sp) categories
    """
    import os
    subj_cats_dict = {}
    subj_cats_dict["hc"] = [subj_num for subj_num in os.listdir(os.path.join(eeg_dir, "hc")) if subj_num.isdigit()]
    subj_cats_dict["sp"] = [subj_num for subj_num in os.listdir(os.path.join(eeg_dir, "sp")) if subj_num.isdigit()]
    return subj_cats_dict
    ## added to new utils
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
    import os
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
# added to new utils

def mat2dict(mat_array):
    '''
    reformat object numpy array returned by scipy.io.loadmat for mat files as dictionaries
    '''
    mat_dict = {field_nm: mat_array[field_nm][:] for field_nm in mat_array.dtype.names}
    return mat_dict
#added to new utils


def get_stims_dict(stim_fnm):
    # NOTE: determined that both stim files have same orig_noisy stim wavs
    import scipy.io as spio
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
#added to new utils
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
    ## added to new utils

def get_full_raw_eeg(eeg_dir, choose_hc_sp:str, subj_num:str, blocks=None):
    import os
    import h5py
    import numpy as np
    """
    eeg_dir: root directory where eeg data is stored
    choose_hc_sp: ("hc" or "sp") sub-directory for healthy control subjects or schizophrenia patients
    subj_num: individual subject number
    blocks: which blocks to get for that subject
    returns:
        subj_eeg: {'block_num': np.ndarray (time x channels) }
    """
    if blocks is None:
        # get all 6 blocks by default
        blocks = [f"B{ii}" for ii in range(1, 6+1)]
    # get raw data hdf5 fnms
    eeg_fnms = [fnm for fnm in os.listdir(os.path.join(eeg_dir, choose_hc_sp, subj_num, "original")) if fnm.endswith('.hdf5')]
    #NOTE: below assumes data for all six blocks present in original folder
    #NOTE: also assumes that block names in order.. for subj 3244 B1 was not present but eeg_fnms_dict assigned keys w block names
    # that were off by one because of it...
    eeg_fnms_dict = {block_num: fnm for block_num, fnm in zip(blocks, eeg_fnms)}
    subj_eeg = {}
    # get raw eeg data

    for block_num in blocks:

        # get eeg 
        eeg_fnm = os.path.join(eeg_dir, choose_hc_sp, subj_num, "original",
                            eeg_fnms_dict[block_num])
        block_file = h5py.File(eeg_fnm) #returns a file; read mode is default
        subj_eeg[block_num] =  np.asarray(block_file['RawData']['Samples'])
        del block_file
    return subj_eeg
# added to new utils


def align_responses(subj_eeg:dict, stim_start_end:dict, stims_dict:dict):
    '''
    subj_eeg: {'block_num': [time x channels] - eeg numpy array}
    stim_start_end {'block_num': {'stim_nm': (onset_time, offset_time)} }
    stims_dict: master stim directory with original stim wav files (noisy and clean)
    '''
    #TODO: add optional padding around on/of time edges 
    # "unwrapping" blocks dim
    import numpy as np
    import pandas as pd

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
# added to new utils

def get_lp_env(env, crit_freq, fs):
    '''
    env: recording envelope
    crit_freq: lowpass filter cutoff f
    fs: recording fs
    '''
    # create lowpass filter
    from scipy import signal
    sos = signal.butter(16, crit_freq, fs=fs, output='sos')
    lp_env = signal.sosfiltfilt(sos, env)
    return lp_env
## added to new utils

def set_thresh(og_rec, lp_env, fs):
    import numpy as np
    import matplotlib.pyplot as plt
    t = np.arange(0, og_rec.size/fs, 1/fs)
    ylim = [0, og_rec.std()*3.0]
    plt.plot(t, og_rec, t, lp_env)
    plt.xlabel('Seconds')
    plt.ylim(ylim)
    plt.show()
    satisfaction = ['y', 'ya', 'sure', 'yes', 'ok', 'why not']
    satisfied = False
    thresh = None
    feelings=None
    while not satisfied:
        thresh = float(input('Choose a threshold.'))
        on_off_times = np.zeros(lp_env.size, dtype=bool)
        on_off_times[1:] = np.diff(lp_env>thresh)
        plt.plot(t, og_rec, t, lp_env)
        plt.xlabel('Seconds')
        plt.hlines(thresh, t[0], t[-1], color='r', label='thresh')
        plt.vlines(t[on_off_times], ylim[0], ylim[1], color='r')
        plt.ylim(ylim)
        plt.show()
        feelings = input('Threshold good enough? y/n').lower()



        
        if feelings in satisfaction:
            garbage_out = False
            while not garbage_out:
                # endpoints might be stupid
                first = ['f', 'first', '1']
                last = ['l', 'last', '2']
                #eg: "first and last" gets rid of both
                discard_endpoints = input('Discard endpoints? (first and/or last)').lower()
                if any([(f in discard_endpoints) for f in first]):
                    # ignore first n thresh crossing
                    #NOTE: must be an int
                    n_bad_starts = int(input('How many false starts? (int)'))
                    on_off_times[np.where(on_off_times)[0][np.arange(n_bad_starts)]] = 0             
                if any([(l in discard_endpoints) for l in last]):
                    # ignore last n thresh crossing
                    n_bad_ends = int(input('How many false starts? (int)'))
                    on_off_times[np.where(on_off_times)[0][-np.arange(n_bad_ends)]] = 0
                # visual confirmation:
                plt.plot(t, og_rec, t, lp_env)
                plt.xlabel('Seconds')
                plt.hlines(thresh, t[0], t[-1], color='r', label='thresh')
                plt.vlines(t[on_off_times], ylim[0], ylim[1], color='r')
                plt.ylim(ylim)
                plt.show()
                feelings = input('All bad crossings out?').lower()
                if feelings in satisfaction:
                    garbage_out = True
                    satisfied = True

    #NOTE: on_off_times is actually bool array with ones where lp envelope crosses
    # threshold, not actually a list of times
    return thresh, on_off_times
# added to new utils

def segment(x, on_off_times, segment_ii=None):
    '''
    x: 1-D array to be split into n subarrays
    on_off_times: 1D bool array with 1's (2n) at onset and offset indices
    segment_ii: if only one segment from on_off_times is desired, choose this one
        (TODO: using to re-slice from original full recording 
        kinda stupid solution might want to change?)
    returns x_segments: list of n 1D arrays from x 
    '''
    import numpy as np
    # assuming we have even number since each onset should have an offset, so check
    # for evenness
    if on_off_times[on_off_times].size %2 != 0:
        raise NotImplementedError('Uneven number of onset/offset pairs')
    # evens -> onsets; odds -> offsets
    start_iis = np.where(on_off_times)[0][::2]
    end_iis = np.where(on_off_times)[0][1::2]
    if segment_ii is not None:
        #only keep one set of start/end times
        start_iis = [start_iis[segment_ii]]
        end_iis = [end_iis[segment_ii]]
    x_segments = [x[start:end] for start, end in zip(start_iis, end_iis)]
    #TODO: how to relate indices back to original eeg recording so we can 
    # slice other eeg channels at correct location?
    return x_segments
#added to new utils

def match_waves(x, y, confidence_lims:list, fs:int):
    '''
    x: long waveform (1d array)
    y: shorter waveform (1d array)

    xcorr wrapper to find where in x y is likely to be
    assumes x is longer than y so that lagging y relative to x gives positive lag values 
    at the lag where the two signals overlap
    '''
    from scipy import signal
    from scipy.stats import pearsonr
    import numpy as np
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
#added to new utils

def get_missing_stim_nms(stim_start_end):
    '''
    stim_start_end: {'block': {'stim_nm': (start, end, confidence)}}
    '''
    missing_stims = []
    for block in stim_start_end.keys():
        for stim_nm, (start, end, _) in stim_start_end[block].items():
            if start is not None:
                missing_stims.append(stim_nm)
    return missing_stims
# added to new utils

def get_time_between_stims(fs_eeg, subj_eeg, stim_start_end):
    # get time between stims in each block
    # NOTE: probably much more efficient to do w np.diff once missing stims removed?
    time_between = []
    for block in subj_eeg:
        prev_end = 0.0
        block_time = make_time_vector(fs_eeg, subj_eeg[block].shape[0])
        for _, (start, end, _) in stim_start_end[block].items():
            if start is not None:
                start_time = block_time[start]
                time_between.append(start_time - prev_end)
                prev_end = block_time[end]
    return time_between
# added to new utils

def make_time_vector(fs, nsamples, start_time=0):
    import numpy as np
    return np.arange(start_time, nsamples/fs, 1/fs)
#added to new utils

def plot_waveform(x:list, fs:list, labels=[None], **kwargs):
    '''
    wrapper around subplots to plot multiple waveforms
    NOTE: implicitly assumes nrows and ncols correspond w number of waveforms given
    which we may want to change
    fs: sampling frequency as float or list if each waveform sampled at different rate
    x: 1d array or list of arrays if multiple waveforms
    label: MUST be a list of strings (or None), one per x
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axs = plt.subplots(**kwargs)
    #TODO: when plotting on same axes, sort waveforms so they are easy to look at
    if any(labels) and len(labels) != len(x):
        #TODO: try except might make more sense here but idk how to define exception properly still
        raise NotImplementedError(f'''{len(labels)} number of labels doesnt 
                                  match number of waveforms ({len(x)})''')

    if isinstance(axs, plt.Axes):
        # plot waveforms on same axes
        if isinstance(x, list) and isinstance(fs, list):
            print('a')
            # different sampling frequencies
            for xi, fsi, label in zip(x, fs, labels):
                t = make_time_vector(fsi, xi.size)
                axs.plot(t, xi, label=label)
        elif isinstance(x, list) and not isinstance(fs, list):
            print('b')
            # same sampling frequencies
            for xi, label in zip(x, labels):
                t = make_time_vector(fs, xi.size)
                axs.plot(t, xi, label=label)
        else:
            print('c')
            # single waveform
            t = make_time_vector(fs, x.size)
            axs.plot(t, x, label=labels)
    else:
        # plot waveforms on separate axes
        if isinstance(x, list) and isinstance(fs, list):
            # different sampling frequencies
            for xi, fsi, ax, label in zip(x, fs, axs.flatten(), labels):
                t = make_time_vector(fsi, xi.size)
                print(xi.shape)
                ax.plot(t, xi, label=label)
                ax.legend()
        elif isinstance(x, list) and not isinstance(fs, list):
            # same sampling frequencies
            for xi, ax, label in zip(x, axs.flatten(), labels):
                t = make_time_vector(fs, xi.size)
                ax.plot(t, xi, label=label)
                ax.legend()
        else:
            raise NotImplementedError(f'x needs to be a list of waveforms when multiple axes.')

        
    # plt.tight_layout()
    # plt.show()
    return fig, axs
# added to new utils

def get_timestamps(subj_eeg, eeg_dir, subj_num, choose_hc_sp, stims_dict, blocks):
    import numpy as np
    import scipy.io as spio
    import os
    from scipy import signal
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
#added to new utils
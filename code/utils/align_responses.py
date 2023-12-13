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
import numpy as np
def segment(x, on_off_times, segment_ii=None):
    '''
    x: 1-D array to be split into n subarrays
    on_off_times: 1D bool array with 1's (2n) at onset and offset indices
    segment_ii: if only one segment from on_off_times is desired, choose this one
        (TODO: using to re-slice from original full recording 
        kinda stupid solution might want to change?)
    returns x_segments: list of n 1D arrays from x 
    '''
    
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
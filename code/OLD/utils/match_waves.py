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
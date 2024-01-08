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

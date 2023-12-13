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

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

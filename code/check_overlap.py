def check_overlap(on_off_times:dict):
    import numpy as np
    result = {}
    for block in on_off_times:
        result[block] = np.zeros((len(block.keys()), 2))

        for ii, stim_nm, times in enumerate(block.items()):
            result[block][ii] = np.array(times)
    #TODO: how to figure out if any of the times actually overlap?

    return result
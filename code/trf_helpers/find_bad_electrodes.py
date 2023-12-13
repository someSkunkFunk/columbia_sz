import numpy as np
def find_bad_electrodes(subj_data, criteria="4std"):
    # NOTE: this will implicitly drop missing trials as well
    if criteria == "4std":
        # reject electrodes whose std is over 4 times greater than median std
        # aggregate all stims into one array with dims [time x electrodes]
        R = np.concatenate([subj_data.dropna()['eeg'].loc[subj_data.dropna()['stim_nms']==nm].iloc[0] 
                            for nm in subj_data.dropna()["stim_nms"]])
        stds = np.std(R, axis=0)
        cutoff = 4*np.median(stds)
        outlier_idx = stds > cutoff

    return outlier_idx

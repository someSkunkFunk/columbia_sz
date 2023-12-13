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

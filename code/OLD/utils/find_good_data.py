import os
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
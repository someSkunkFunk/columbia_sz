import os
import h5py
import numpy as np
def get_full_raw_eeg(eeg_dir, choose_hc_sp:str, subj_num:str, blocks=None):
    
    """
    eeg_dir: root directory where eeg data is stored
    choose_hc_sp: ("hc" or "sp") sub-directory for healthy control subjects or schizophrenia patients
    subj_num: individual subject number
    blocks: which blocks to get for that subject
    returns:
        subj_eeg: {'block_num': np.ndarray (time x channels) }
    """
    if blocks is None:
        # get all 6 blocks by default
        blocks = [f"B{ii}" for ii in range(1, 6+1)]
    # get raw data hdf5 fnms
    eeg_fnms = [fnm for fnm in os.listdir(os.path.join(eeg_dir, choose_hc_sp, subj_num, "original")) if fnm.endswith('.hdf5')]
    #NOTE: below assumes data for all six blocks present in original folder
    #NOTE: also assumes that block names in order.. for subj 3244 B1 was not present but eeg_fnms_dict assigned keys w block names
    # that were off by one because of it...
    eeg_fnms_dict = {block_num: fnm for block_num, fnm in zip(blocks, eeg_fnms)}
    subj_eeg = {}
    # get raw eeg data

    for block_num in blocks:

        # get eeg 
        eeg_fnm = os.path.join(eeg_dir, choose_hc_sp, subj_num, "original",
                            eeg_fnms_dict[block_num])
        block_file = h5py.File(eeg_fnm) #returns a file; read mode is default
        subj_eeg[block_num] =  np.asarray(block_file['RawData']['Samples'])
        del block_file
    return subj_eeg
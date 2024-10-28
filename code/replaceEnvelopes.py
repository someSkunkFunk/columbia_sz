#%%
# script for replacing time-aligned stimuli in subj preprocessed data
# NOTE: will re-saave preprocessed data in new format onto new directory to 
# avoid confusion/compatability issues with old analysis code
import utils
import os
import numpy as np
import pickle as pkl
from mat4py import loadmat
import h5py

#%%
# define helpers
def replace_stimuli(subj_data):
    '''
    helper function for replacing stimuli which have already been time-aligned
    '''
    for trial, (stim_ids,stim,resp) in subj_data.items():
        print(f"{trial}...")
        stim_ids=[s.strip('.wav') for s in stim_ids]
        idx=np.where(np.isin(stim_info['ID'],stim_ids))[0]
        temp_stims=[new_envelopes['envelopes'][ii] for ii in idx]
        new_stim=np.concatenate(temp_stims)
        del temp_stims, idx
        if durations_check(new_stim,resp):
            # if close enough in durations, drop extra samples from whichever is longer
            n_dur=np.min([new_stim.shape[0],resp.shape[0]])
            new_stim=new_stim[:n_dur]
            resp=resp[:n_dur,:]
            subj_data[trial]=(stim_ids,new_stim,resp)
        else:
            raise NotImplementedError(f'''
            durations too different at trial {trial} -
            stim,resp shapes: {new_stim.shape}, {resp.shape}
            ''')
    print('all trials replaced with new stimuli.')


def durations_check(x,y):
    # assumes same sampling rates and time dimension is first
    #maximum number of samples x/y durations can be off by and be considered correct
    n_tol=5
    if np.abs(x.shape[0]-y.shape[0])<=n_tol:
        durations_equal=True
    else:

        durations_equal=False
    return durations_equal

def load_raw_eeg(subj_num:str, blocks=None):
    
    """
    copied from utils.get_full_raw_eeg... don't want to break old preprocessing code but also
    the way this function was originally written was stupid so I wanted to update it for future use
    ASSUMES DATA FOR ALL 6 BLOCKS IS PRESENT IN RAW FOLDER
    raw_dir: root directory where raw eeg data is stored
    subj_cat: ("hc" or "sp") sub-directory for healthy control subjects or schizophrenia patients
    subj_num: individual subject number
    blocks: which blocks to get for that subject
    returns:
        subj_eeg: {'block_num': np.ndarray (time x channels) }
    """
    raw_dir=os.path.join("..","eeg_data","raw")
    subj_cat=utils.get_subj_cat(subj_num)
    if blocks is None:
        # get all 6 blocks by default
        blocks = [f"B{ii}" for ii in range(1, 6+1)]
    # get raw data hdf5 fnms
    eeg_fnms = [fnm for fnm in os.listdir(os.path.join(raw_dir, subj_cat, subj_num, "original")) if fnm.endswith('.hdf5')]
    #NOTE LISTDIR DOESN'T GIVE THEM IN ORDER, sorting assumes block data files names consistent acrosss subjects
    eeg_fnms.sort()
    #NOTE: below assumes data for all six blocks present in original folder
    #NOTE: also assumes that block names in order.. for subj 3244 B1 was not present but eeg_fnms_dict assigned keys w block names
    # that were off by one because of it...
    # eeg_fnms_dict = {block_num: fnm for block_num, fnm in zip(blocks, eeg_fnms)}
    
    subj_eeg = {}
    # get raw eeg data

    for block_num in blocks:
        # match block to correct data
        #note assumes block is capitalized and at start of hdf5 files
        eeg_fl=list(filter(lambda x: x.startswith(block_num),eeg_fnms))        
        # check that at least one data file is found
        if any(eeg_fl):
            # check only one file contained:
            if len(eeg_fl)!=1:
                raise NotImplementedError('there may be too many matching files here')
            eeg_fnm=eeg_fl[0]
            eeg_pth = os.path.join(raw_dir, subj_cat, subj_num, "original",
                                eeg_fnm)
            # get eeg 
            with h5py.File(eeg_pth,'r') as block_file:
                subj_eeg[block_num]=np.asarray(block_file['RawData']['Samples'])
        else:
            print(f"{block_num} EEG data not found!")
    return subj_eeg
#%%
#  main script
new_stim_dir=os.path.join("..","data","preprocessed_eeg")
noisy_or_clean='clean'
subj_num='3316'
evnt_thresh=0
thresh_dir=f"thresh_{round(evnt_thresh*1000):03}"
old_prep_data_dir=os.path.join("..","eeg_data",'preprocessed_decimate',thresh_dir)
#TODO: update load_preprocessed
subj_data=utils.load_preprocessed(subj_num,eeg_dir=old_prep_data_dir)
stim_info_pkl_path=os.path.join("..","stimuli","stim_info.pkl")
with open(stim_info_pkl_path,'rb') as file:
    stim_info=pkl.load(file)
envelopes_path=os.path.join("..","stimuli",f"{noisy_or_clean}Envs.pkl")
with open(envelopes_path,'rb') as file:
    new_envelopes=pkl.load(file)
trials=list(subj_data.keys())

# load evnt timestamps
#TODO: make evnt timestamps loading function a util
# check if save directory exists, else make one
## find evnt timestamps
timestamps_path=os.path.join("..","eeg_data","timestamps",f"evnt_{subj_num}.mat")
evnt=loadmat(timestamps_path)['evnt']
# going based off fact that our preprocessing code didn't remove any pauses due to start times 
# happening before previous stimulus ended, going to ignore that possibility here
# also using fact that start times monotonically increasing within a block to assume time relative
# to block start is used 
#%%
raw_eeg=load_raw_eeg(subj_num)
#%%
# look at random trial to ensure being replaced properly
n_trial=50
inspected_trial=subj_data[trials[n_trial]]
print(f"{n_trial} stim,resp shape before: {inspected_trial[1].shape},{inspected_trial[2].shape}")
replace_stimuli(subj_data)
print(f"{n_trial} stim,resp shape after: {inspected_trial[1].shape},{inspected_trial[2].shape}")


#%%
# stim_info=utils.get_stims_dict()
# # save for future use, only need to run once (ideally)
# with open(stim_info_pkl_path, 'wb') as file:
#     pkl.dump(stim_info, file)

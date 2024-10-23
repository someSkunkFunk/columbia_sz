#%%
# NOTE: will re-saave preprocessed data in new format onto new directory to 
# avoid confusion/compatability issues with old analysis code
import utils
import os
import numpy as np
import pickle as pkl
#%%
#TODO: need to re-align stimuli envelopes with responses... 
# I think we can take previously aligned data and it should have the stimuli
#  names which we can use to match and swap the new features with....? 
#%%
new_stim_dir=os.path.join("..","data","timeAligned")
noisy_or_clean='clean'
subj_num='3316'
evnt_thresh='000'
thresh_dir=f"thresh_{evnt_thresh}" # should be three digit number representing decimals to third place
prep_data_dir=os.path.join("..","eeg_data",'preprocessed_decimate',thresh_dir)
subj_data=utils.load_preprocessed(subj_num,eeg_dir=prep_data_dir)
stim_info_pkl_path=os.path.join("..","stimuli","stim_info.pkl")
with open(stim_info_pkl_path,'rb') as file:
    stim_info=pkl.load(file)
envelopes_path=os.path.join("..","stimuli",f"{noisy_or_clean}Envs.pkl")
with open(envelopes_path,'rb') as file:
    new_envelopes=pkl.load(file)
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
            n_dur=np.min(new_stim.shape[0],resp.shape[0])
            new_stim=new_stim[:n_dur]
            resp=resp[:ndur,:]
            subj_data[trial]=(stim_ids,new_stim,resp)
        else:
            raise NotImplementedError(f'durations too different at trial {trial}')
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
#%%
#  main script
trials=list(subj_data.keys())
# look at random trial to ensure being replaced properly
n_trial=50
inspected_trial=subj_data[trials[n_trial]]
print(f"{n_trial} stim,resp shape before: {inspected_trial[1].shape},{inspected_trial[2].shape}")
replace_stimuli(subj_data)
print(f"{n_trial} stim,resp shape after: {inspected_trial[1].shape},{inspected_trial[2].shape}")


#%%
stim_info=utils.get_stims_dict()
# save for future use, only need to run once (ideally)
with open(stim_info_pkl_path, 'wb') as file:
    pkl.dump(stim_info, file)

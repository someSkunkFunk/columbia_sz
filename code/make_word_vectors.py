#script for converting textgrid file information into phoneme and word vectors
#%%
# INIT
import textgrids
import os
import utils

#%%
# EXEC
stim_fl_path=os.path.join("..","eeg_data","stim_info.mat")
stims_dict=utils.get_stims_dict(stim_fl_path)
# strip prefix and suffix in stim names to match textgrid file names
#%%
stim_names=[n.replace('./Sounds/','') for n in stims_dict['Name']]
#note pretty sure this means the textgrids have time in 16kHz 
# fs instead of original audio fs
stim_names=[n.replace('_16K_NM.wav','') for n in stim_names]

textgrids_path=os.path.join("..","eeg_data","textgrids")
textgrids_fnms=os.listdir(textgrids_path)
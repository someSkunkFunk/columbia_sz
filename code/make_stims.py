#%%
# INIT
from trf_helpers import get_stim_envs 
from utils import get_stims_dict
import os
import pickle
stims_dict=get_stims_dict()


clean_or_noisy='clean'
fs=100 # for trf analysis
lp_f=49 # low pass frequency
norm=True


#%%
#MAIN SCRIPT
stim_envs=get_stim_envs(stims_dict,clean_or_noisy,fs_output=fs,f_lp=lp_f,norm=norm) #NOTE: lowpass 49 cutoff by default and filter order is 1
if norm:
    norm_str="_normalized"
else:
    norm_str=""
save_fnm=f"stim_envs_{str(lp_f)}hz_{clean_or_noisy}{norm_str}.pkl"
save_pth=os.path.join("..","eeg_data",save_fnm)

with open(save_pth,'wb') as fl:
    pickle.dump(stim_envs,fl)
    print(f"saved stim_envs to {save_pth}")
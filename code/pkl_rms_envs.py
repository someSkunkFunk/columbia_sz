# dummy script to save rms envelopes for easier loading
#%%
# INIT
import utils
import pickle
import os
#%%
# EXEC
noisy_or_clean="clean"
envs=utils.load_matlab_envs(noisy_or_clean)
save_path=os.path.join("..","eeg_data",f"{noisy_or_clean}_rms_envelopes.pkl")

with open(save_path,'wb') as file:
    pickle.dump(envs,file)
    
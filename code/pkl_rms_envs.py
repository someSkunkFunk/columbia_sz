# dummy script to save rms envelopes for easier loading
#%%
# INIT
import utils
import pickle
import os
#%%
# EXEC
envs=utils.load_matlab_envs('clean')
save_path=os.path.join("..","eeg_data","rms_envelopes.pkl")

with open(save_path,'wb') as file:
    pickle.dump(envs,file)
    
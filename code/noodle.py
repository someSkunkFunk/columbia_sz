#%%
import utils
import scipy.io as spio
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

#%%
subj_num="3326"
# get test subject raw data
#TODO: update evnt_path so it reflects test subject

raw_dir=os.path.join("..", "eeg_data", "raw")
subj_cat=utils.get_subj_cat(subj_num)
subj_eeg=utils.get_full_raw_eeg(raw_dir,subj_cat,subj_num)
#TODO: load stims dict, compare a few stims visually for this subject, evaluate pearsonr values 
# (to see if they jive with reported confidence or not)
# then try and evaluate for all the subjects/stimuli somehow?
stims_pth=os.path.join("..", "eeg_data", "stim_info.mat")
stims_dict=utils.get_stims_dict(stims_pth)
#%%
# select stim to plot
stim_index=0
fs_eeg=2400
#TODO: get block from selected_stim_nm
fs_audio=stims_dict['fs'][0]

#which timestamps to use: evnt or mine
choose_timestamps="mine"
#use timestamps provided in evnt structure
if choose_timestamps=="evnt":
    # NOTE: not sure if names will certainly be in order
    evnt_path=os.path.join("..", "eeg_data", "timestamps", f"evnt_{subj_num}.mat" )
    evnt_mat=spio.loadmat(evnt_path)
    # returns dict for some reason, which mat2dict doesnt like
    evnt=evnt_mat['evnt']
    mystery=utils.mat2dict(evnt)
    # get using weird indexing because object ndarray
    indexed_stim_nm=mystery['name'][0, stim_index][0]
    block=indexed_stim_nm[:3].capitalize().replace('0','')
    sync_position=mystery['syncPosition'][0,stim_index][0,0]
    

elif choose_timestamps=="mine":
    tmstmps_pth=os.path.join("..","eeg_data","timestamps",subj_cat,subj_num,"timestamps.pkl")
    with open(tmstmps_pth, 'rb') as f:
        timestamps=pickle.load(f)
    # flatten blocks so we can match flattened index
    all_nms=[]
    all_stmps=[]
    for _, nms_n_stamps in timestamps.items():
        for nm, stmps in nms_n_stamps.items():
            all_nms.append(nm)
            all_stmps.append(stmps)
    indexed_stim_nm=all_nms[stim_index]
    block=indexed_stim_nm[:3].capitalize().replace('0','')
    sync_positions=all_stmps[stim_index]

stim_wav=utils.get_stim_wav(stims_dict,indexed_stim_nm,'clean')
# get eeg sound recording 
stim_dur=(stim_wav.size-1)/fs_audio
if choose_timestamps=="evnt":
    eeg_wav=subj_eeg[block][sync_position:sync_position+int(stim_dur*fs_eeg+1),-1]
elif choose_timestamps=="mine":
    eeg_wav=subj_eeg[block][sync_positions[0]:sync_positions[1],-1]
#%% 
# plot    


plt.figure()
plt.plot(stim_wav)
plt.title('Stim Wav')
plt.show()
plt.figure()
plt.plot(eeg_wav)
plt.title('EEG WAV')
plt.show()
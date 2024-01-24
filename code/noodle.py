#%%
import utils
import scipy.io as spio
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

#%%
#select subject
subj_num="3318"
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
# segment eeg using both timestamps
stim_index=400
fs_eeg=2400
#TODO: get block from selected_stim_nm
fs_audio=stims_dict['fs'][0]

#use timestamps provided in evnt structure
# NOTE: not sure if names will certainly be in order
evnt_path=os.path.join("..", "eeg_data", "timestamps", f"evnt_{subj_num}.mat" )
evnt_mat=spio.loadmat(evnt_path)
# returns dict for some reason, which mat2dict doesnt like
evnt=evnt_mat['evnt']
mystery=utils.mat2dict(evnt)
# get using weird indexing because object ndarray
stim_nm_evnt=mystery['name'][0, stim_index][0]
block_evnt=stim_nm_evnt[:3].capitalize().replace('0','')
sync_position=mystery['syncPosition'][0,stim_index][0,0]

# use my timestamps
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
stim_nm_stmps=all_nms[stim_index]
block_stmps=stim_nm_stmps[:3].capitalize().replace('0','')
sync_positions=all_stmps[stim_index]



# get eeg segments and corresponding stim wav

# i suspect evnt timestamps are off by some constant
shift=12000
if stim_nm_stmps!=stim_nm_evnt:
    raise NotImplementedError('stim names dont match')

#below assumes names match
stim_wav=utils.get_stim_wav(stims_dict,stim_nm_stmps,'clean')
stim_dur=(stim_wav.size-1)/fs_audio
eeg_wav_evnt=subj_eeg[block_evnt][shift+sync_position:shift+sync_position+int(stim_dur*fs_eeg+1),-1]
eeg_wav_stmps=subj_eeg[block_stmps][sync_positions[0]:sync_positions[1],-1]
#%% 
# plot    


plt.figure()
plt.plot(np.arange(0,stim_wav.size)/fs_audio, stim_wav)
plt.title('Stim Wav')
plt.show()
plt.figure()
plt.plot(np.arange(0,eeg_wav_stmps.size)/fs_eeg, eeg_wav_stmps)
plt.title('my EEG WAV')
plt.show()
plt.figure()
plt.plot(np.arange(0,eeg_wav_evnt.size)/fs_eeg,eeg_wav_evnt)
plt.title('EEG WAV evnt')
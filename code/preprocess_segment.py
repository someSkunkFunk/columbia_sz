# Already have stim timestamps for each stim in some subjects; for those
# we just preprocess and segment
# for subjects without timestamps, first get the timestamps, then preprocess and segment
#%%
# imports, etc.
import pickle
import scipy.io as spio
import numpy as np

import os
# import sounddevice as sd # note not available via conda....? not sure I'll need anyway so ignore for now
from scipy import signal
import matplotlib.pyplot as plt
import utils

# specify fl paths assumes running from code as pwd
eeg_dir=os.path.join("..", "eeg_data")
# stim_fnm = "master_stim_file_schiz_studybis.mat" # note this is original fnm from box, we changed to just stim_info.mat
stim_fnm="stim_info.mat"
stim_fl_path=os.path.join(eeg_dir, "stim_info.mat")
stims_dict=utils.get_stims_dict(stim_fl_path)
fs_audio=stims_dict['fs'][0] # 11025 foriginally
# fs_audio=16000 #just trying it out cuz nothing else works
fs_eeg=2400 #trie d2kHz didn't help
fs_trf=100 # Hz, downsampling frequency for trf analysis
n_blocks = 6
blocks = [f"B{ii}" for ii in range(1, n_blocks+1)]#NOTE: this is kinda unnecesary and I wanna remove it but focusing on bigger problem rn
#%%
# setup
# bash script vars
# 
###########################################################################################
if "subj_num" in os.environ: 
    subj_num=os.environ["subj_num"]
    which_stmps=os.environ["which_stmps"] #xcorr or evnt
    which_xcorr=os.environ["which_xcorr"]

    bool_dict={"true":True,"false":False}
    just_stmp=bool_dict[os.environ["just_stmp"].lower()]
    print(f"just_stamp translated into: {just_stmp}")
#####################################################################################
#manual vars
#####################################################################################
else:
    print("using manually inputted vars")
    subj_num="3253"
    which_stmps="xcorr"
    which_xcorr="wavs"
    just_stmp=False
    # noisy_or_clean="clean" #NOTE: clean is default and setting them here does nothing
##################################################################################
confidence_lims=[0.80, 0.40]
#print max, min pearsonr correlation thresholds
print(f'minimum pearsonr threshold: {min(confidence_lims)}')
timestamps_bad=False #we trust the stamps now, but there may be a bug in sementation
# determine filter params applied to EEG before segmentation 
# NOTE: different from filter lims used in timestamp detection algo (!)
filt_band_lims=[1.0, 15] #Hz; highpass, lowpass cutoffs
filt_o=3 # order of filter (effective order x2 of this since using zero-phase)
processed_dir_path=os.path.join(eeg_dir, f"preprocessed_{which_stmps}") #directory where processed data goes
subj_cat=utils.get_subj_cat(subj_num) #note: checked get_subj_cat, should be fine
raw_dir=os.path.join(eeg_dir,"raw")
print(f"Fetching data for {subj_num,subj_cat}")
subj_eeg=utils.get_full_raw_eeg(raw_dir,subj_cat,subj_num,blocks=blocks)
#%%
# find timestamps
if which_stmps=="xcorr":
    # Find timestamps using xcorr algo
    # check if save directory exists, else make one
    save_path=os.path.join(processed_dir_path,subj_cat,subj_num)
    if not os.path.isdir(save_path):
        print(f"preprocessed dir for {subj_num, subj_cat} not found, creating one.")
        os.makedirs(save_path, exist_ok=True)
    # check if timestamps fl exists already
    timestamps_path=os.path.join("..","eeg_data","timestamps",subj_cat,subj_num,
                                f"{which_xcorr}_timestamps.pkl")
    if os.path.exists(timestamps_path) and not timestamps_bad:
        # if already have timestamps, load from pkl:
        print(f"{subj_num, subj_cat} already has timestamps, loading from pkl.")
        with open(timestamps_path, 'rb') as pkl_fl: 
            timestamps = pickle.load(pkl_fl)
    else:
        print(f"Generating timestamps for {subj_num, subj_cat} ...")
        # get timestamps
        timestamps=utils.get_timestamps(subj_eeg,raw_dir,subj_num,
                                        subj_cat,stims_dict,blocks,which_xcorr,
                                        confidence_lims)
        # check resulting times
        total_soundtime=0
        missing_stims_list=[]
        for block in timestamps:
            block_sound_time=0
            for stim_nm, (start, end, confidence) in timestamps[block].items():
                if all([start,end,confidence]):
                    block_sound_time+=(end-start-1)/fs_eeg
                else:
                    missing_stims_list.append(stim_nm)
            print(f"in block {block}, total sound time is {block_sound_time:.3f} s.")
            total_soundtime+=block_sound_time
        print(f"total sound time: {total_soundtime:.3f} s.")
        print(f"missing stims:\n{len(missing_stims_list)}")
        #  save stim timestamps
        with open(timestamps_path, 'wb') as f:
            print(f"saving timestamps for {subj_num}")
            pickle.dump(timestamps, f)
if which_stmps=="evnt":
    # load evnt timestamps
    #TODO: make evnt timestamps loading function a util
    # check if save directory exists, else make one
    save_path = os.path.join(processed_dir_path, subj_cat, subj_num)
    if not os.path.isdir(save_path):
        print(f"preprocessed dir for {subj_num, subj_cat} not found, creating one.")
        os.makedirs(save_path, exist_ok=True)
    # find evnt timestamps
    timestamps_path=os.path.join("..","eeg_data","timestamps",f"evnt_{subj_num}.mat")
    evnt_mat=spio.loadmat(timestamps_path)
    # returns dict for some reason, which mat2dict doesnt like
    evnt=evnt_mat['evnt']
    evnt=utils.mat2dict(evnt)
#%%
# preprocess each block separately
if not just_stmp:
    timestamps_ds = {}
    print(f"Starting preprocessing {subj_num, subj_cat}")
    for block, raw_data in subj_eeg.items():
        #TODO: modify preprocessing block so that audio recording channel not filtered and can be used to check for alignment issues
        print(f"block: {block}")
        timestamps_ds[block] = {}
        # filter and resample
        if fs_eeg / 2 <= fs_trf:
            raise NotImplementedError("Nyquist") 
        sos=signal.butter(filt_o,filt_band_lims,btype='bandpass',
                          output='sos',fs=fs_eeg)
        filt_eeg=signal.sosfiltfilt(sos,raw_data[:,:62],axis=0)
        audio_rec=raw_data[:,-1] # leave unperturbed for alignment checking
        # get number of samples in downsampled waveform
        num_ds=int(np.floor((filt_eeg.shape[0]-1)*(fs_trf/fs_eeg)))
        # downsample eeg
        # NOTE: audio_rec not downsampled
        subj_eeg[block]=(signal.resample(filt_eeg,num_ds,axis=0),audio_rec)
        # downsample timestamps
        for stim_nm, (start, end, confidence) in timestamps[block].items():
            if all([start,end,confidence]):
                
                s_ds=int(np.floor(start*(fs_trf/fs_eeg)))
                e_ds=int(np.floor(end*(fs_trf/fs_eeg))) #NOTE: off by one error maybe?
                timestamps_ds[block][stim_nm]=(s_ds, e_ds, confidence)
            else:
                timestamps_ds[block][stim_nm]=(None, None, confidence)
    #%
    # align downsampled eeg using ds timestamps
    print(f"Preprocessing done for {subj_num, subj_cat}. algining and segmenting eeg")
    subj_data = utils.align_responses(subj_eeg, (timestamps, timestamps_ds), 
                                      audio_rec, stims_dict)
    subj_data['fs'] = fs_trf
    print("subj_data before pickling:")
    print(subj_data.head())
    print(f'saving to: {save_path}')
    subj_data.to_pickle(os.path.join(save_path, f"{which_xcorr}_aligned_resp.pkl"))
    print(f"{subj_num, subj_cat} preprocessing and segmentation complete.")
    # break


 
  
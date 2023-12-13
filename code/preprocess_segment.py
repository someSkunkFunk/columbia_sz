# Already have stim timestamps for each stim in some subjects; for those
# we just preprocess and segment
# for subjects without timestamps, first get the timestamps, then preprocess and segment
#%%
# imports
%matplotlib widget
import pickle
import scipy.io as spio
import numpy as np
import os
import sounddevice as sd
import h5py
from scipy import signal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import utils
#%% 
# Get stim information file and eeg_dir
stim_fnm = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\Matlab_scripts EEG\EEG_Study\master_stim_file_schiz_studybis.mat"
stims_dict = utils.get_stims_dict(stim_fnm)
fs_audio = stims_dict['fs'][0] # 11025 foriginally
fs_eeg = 2400
eeg_dir = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\EEG For Emeline\Schizophrenia EEG Study\Subject Data and Results"

n_blocks = 6
blocks = [f"B{ii}" for ii in range(1, n_blocks+1)]

subj_cats_dict = utils.get_subj_cat(eeg_dir)

#%%
# get subjs with raw data and stim order files
#NOTE: not sure if those without stim order files actually need or not since order might be the same?

# returns tuples: ("subj_num", "hdf5_blocks_found", "stimOrderMats_blocks_found")
good_hc_subjs, good_sp_subjs = utils.find_good_data(eeg_dir)
#NOTE: some sp subjs missing eeg fls for some blocks
#%%
# get timestamps, then preprocess
filt_band_lims = [0.5, 40] #Hz; highpass, lowpass cutoffs
filt_o = 3 # order of filter (effective order x2 of this since using zero-phase)
fs_trf = 100 # Hz, downsample frequency for trf analysis
redoing = True # if gotta fix existing preprocessed files due to big goofin'
processed_dir = r"C:\Users\ninet\OneDrive\Desktop\LALOR LAB\PYTHON\Columbia Sz\eeg_data" #directory where processed data goes
# all_good_subjs = good_hc_subjs + good_sp_subjs
for (subj_num, hdf5_blocks, mat_blocks) in all_good_subjs:
    if hdf5_blocks != 'B1B2B3B4B5B6' or mat_blocks != 'B1B2B3B4B5B6':
        # skip subject if some files missing
        # NOTE: had "and" before.. so if only one kind of file was missing (as is the case for subj 3244) still plowed ahead (erroneously)
        continue 
    else:
        if subj_num in subj_cats_dict['sp']:
            choose_hc_sp = "sp" # healthy control ("hc") or schizophrenia patients ("sp")
            subj_eeg = utils.get_full_raw_eeg(eeg_dir, choose_hc_sp, subj_num, blocks=blocks)
        elif subj_num in subj_cats_dict['hc']:
            choose_hc_sp = "hc" # healthy control ("hc") or schizophrenia patients ("sp")
            subj_eeg = utils.get_full_raw_eeg(eeg_dir, choose_hc_sp, subj_num, blocks=blocks)
        else:
            raise NotImplementedError('subj needs to be hc or sp')
        save_loc = os.path.join(processed_dir, choose_hc_sp, subj_num)
        # check if directory exists, else make one
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
        # check if timestamps fl exists already
        stim_start_end_pkl = os.path.join(save_loc, f"{subj_num}_stim_start_end.pkl")
        if os.path.exists(stim_start_end_pkl):
            # if already have timestamps, load from pkl:
            print(f"{subj_num, choose_hc_sp} already has timestamps, loading from pkl...")
            with open(stim_start_end_pkl, 'rb') as pkl_fl: 
                stim_start_end = pickle.load(pkl_fl)
        else:
            print(f"finding timestamps for {subj_num, choose_hc_sp} ...")
            # get timestamps
            stim_start_end = utils.get_timestamps(subj_eeg, eeg_dir, subj_num, choose_hc_sp, stims_dict, blocks)
            #  save stim timestamps
            with open(os.path.join(save_loc, subj_num+'_stim_start_end.pkl'), 'wb') as f:
                pickle.dump(stim_start_end, f)
        # preprocess each block separately
        if os.path.exists(os.path.join(save_loc, subj_num+"_prepr_data.pkl")) and not redoing:
            continue
        elif os.path.exists(os.path.join(save_loc, subj_num+"_prepr_data.pkl")) and redoing:
            print(f'deleting existing preprocessed file for subj {subj_num}')
            os.remove(os.path.join(save_loc, subj_num+"_prepr_data.pkl"))
            print(f're-slicing time stamps for {subj_num}')
            stim_start_end_ds = {}
            print(f"Start: preprocessing for {subj_num, choose_hc_sp}")
            for block, raw_eeg in subj_eeg.items():
                print(f"start: {block}")
                stim_start_end_ds[block] = {}
                # filter and resample
                if fs_eeg / 2 <= fs_trf:
                    raise NotImplementedError("Nyquist") 
                sos = signal.butter(filt_o, filt_band_lims, btype='bandpass', output='sos', fs=fs_eeg)
                raw_eeg = signal.sosfiltfilt(sos, raw_eeg, axis=0)
                ds_factor = int(np.floor((raw_eeg.shape[0]-1)*(fs_trf/fs_eeg)))
                subj_eeg[block] = signal.resample(raw_eeg, ds_factor, axis=0)
                # resample timestamps
                for stim_nm, (start, end, confidence) in stim_start_end[block].items():
                    if start is not None:
                        s_ds = int(np.floor(start*(fs_trf/fs_eeg)))
                        e_ds = int(np.floor(end*(fs_trf/fs_eeg))) #NOTE: off by one error maybe?
                        stim_start_end_ds[block][stim_nm] = (s_ds, e_ds, confidence)
                    else:
                        stim_start_end_ds[block][stim_nm] = (None, None, confidence)
            # align downsampled eeg using ds timestamps
            print(f"Preprocessing done for {subj_num, choose_hc_sp}. algining and slicing eeg")
            #TODO: resulting pkl file has nothing... probably align responses doing something weird?
            subj_data = utils.align_responses(subj_eeg, stim_start_end_ds, stims_dict)
            subj_data['fs'] = fs_trf
            print("subj_data before pickling:")
            print(subj_data.head())
            # print("shape of eeg response for first stim:", subj_data['eeg'][0].shape)
            subj_data.to_pickle(os.path.join(save_loc, subj_num+"_prepr_data.pkl"))
            print(f"{subj_num, choose_hc_sp} preprocessing and segmentation complete.")
        else:
            stim_start_end_ds = {}
            print(f"Start: preprocessing for {subj_num, choose_hc_sp}")
            for block, raw_eeg in subj_eeg.items():
                print(f"start: {block}")
                stim_start_end_ds[block] = {}
                # filter and resample
                if fs_eeg / 2 <= fs_trf:
                    raise NotImplementedError("Nyquist") 
                sos = signal.butter(filt_o, filt_band_lims, btype='bandpass', output='sos', fs=fs_eeg)
                raw_eeg = signal.sosfiltfilt(sos, raw_eeg, axis=0)
                ds_factor = int(np.floor((raw_eeg.shape[0]-1)*(fs_trf/fs_eeg)))
                subj_eeg[block] = signal.resample(raw_eeg, ds_factor, axis=0)
                # resample timestamps
                for stim_nm, (start, end, confidence) in stim_start_end[block].items():
                    if start is not None:
                        s_ds = int(np.floor(start*(fs_trf/fs_eeg)))
                        e_ds = int(np.floor(end*(fs_trf/fs_eeg))) #NOTE: off by one error maybe?
                        stim_start_end_ds[block][stim_nm] = (s_ds, e_ds, confidence)
                    else:
                        stim_start_end_ds[block][stim_nm] = (None, None, confidence)

            # align downsampled eeg using ds timestamps
            print(f"Preprocessing done for {subj_num, choose_hc_sp}. algining and slicing eeg")
            #TODO: resulting pkl file has nothing... probably align responses doing something weird?
            subj_data = utils.align_responses(subj_eeg, stim_start_end_ds, stims_dict)
            subj_data['fs'] = fs_trf
            print("subj_data before pickling:")
            print(subj_data.head())
            subj_data.to_pickle(os.path.join(save_loc, subj_num+"_prepr_data.pkl"))
            print(f"{subj_num, choose_hc_sp} preprocessing and segmentation complete.")
        # break


 
            

        
# %%
# check result for one subject
#NOTE: visually not very convincing but we had a shitty audio to begin with, then filtered and downsampled it so....
subj_num = "3253"
choose_hc_sp = "hc"
with open(os.path.join(os.getcwd(), choose_hc_sp, subj_num, subj_num+"_prepr_data.pkl"), 'rb') as pkl_fl:
    subj_data = pickle.load(pkl_fl)


#%%
# sort stims by stim align confidence vals
subj_data.sort_values(by='confidence', ascending=True, inplace=True) # sort by confidence vals
#NOTE: ALL CODE BELOW ASSUMES subj_data RESORTED BY CONFIDENCE VALS
#TODO: probably better to have consistent stim ordering across subjects rather than this
#%%
# filter out nans and check if eeg recording duration is always larger than stim duration
stim_durs = [(stims_dict['orig_noisy'][stims_dict['ID']==s_nm[:-4]][0].size - 1)/fs_audio for s_nm in subj_data.loc[subj_data['eeg'].notna(), 'stim_nms']]
response_durs = [(r.size - 1)/fs_eeg for r in subj_data.loc[subj_data['eeg'].notna(), 'eeg_audio']]
print(np.all(response_durs < stim_durs))
#NOTE: so all responses consistently shorter than stims due to stim downsampling during xcorr process
# going to just downsample all the stims then pad responses at the beginning if needed to match stim size (shouldn't need)
#%%
# sort by confidence and visually compare
s_num = 700 # which to plot, in order of confidence
subj_data.sort_values(by='confidence', ascending=True, inplace=True)
stim_nm = subj_data.iloc[s_num]['stim_nms']
stim = utils.get_stim_wav(stims_dict, stim_nm, has_wav=True) # gets noisy by default
t_stim = utils.make_time_vector(fs_audio, stim.size)
recording = subj_data.iloc[s_num]['eeg_audio']
if np.isnan(recording).any():
    print(f"recording is {recording}")
else:
    t_recording = utils.make_time_vector(fs_eeg, recording.size)
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(t_stim, stim)
    ax[0].set_title('OG Stim')
    ax[1].plot(t_recording, recording, label=f'confidence:{subj_data.iloc[s_num].confidence}')
    ax[1].set_title('Recorded audio')
    ax[1].set_xlabel('Seconds')
    plt.legend()
    fig.suptitle(f'{subj_data.iloc[s_num].stim_nms}')
    
#%%
# listen to stimulus
sd.play(stim, fs_audio)
#%%
# listen to recording (from df)
sd.play(recording, fs_trf)

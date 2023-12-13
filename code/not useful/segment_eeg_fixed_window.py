# segment eeg audio recording via xcorr using fixed window
# %%
# interactive plots
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
# Get stim information file
stim_fnm = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\Matlab_scripts EEG\EEG_Study\master_stim_file_schiz_studybis.mat"
stims_dict = utils.get_stims_dict(stim_fnm)
fs_audio = stims_dict['fs'][0] # 11025 foriginally
fs_eeg = 2400
# %% 
# Select Subject, get full eeg dataset
eeg_dir = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\EEG For Emeline\Schizophrenia EEG Study\Subject Data and Results"
choose_hc_sp = "sp" # healthy control ("hc") or schizophrenia patients ("sp")
subj_num = '2782' # has all eeg hdf5s and stim mats AND evnt structure for comparison
n_blocks = 6
blocks = [f"B{ii}" for ii in range(1, n_blocks+1)]


subj_eeg = utils.get_full_raw_eeg(eeg_dir, choose_hc_sp, subj_num, blocks=blocks)
# %%
# segment based on threshold
# NOTE: might still be good idea to do xcorr on envelopes rather than

# waveform?

# split envelope first using lp envelope or not
segment_blocks=False 
which_corr = 'wavs' # 'wavs' or 'envs'
confirm_visually = True # check waveforms
# step size for xcorr window calculation
confidence_lims = [0.90, 0.4] #max, min pearsonr correlation thresholds
lpf_cut = 0.1 # lowpass cutoff for envelope-based segmetation

# store indices for each block {"block_num":{"stim_nm": (start, end, rconfidence)}}
stim_start_end = {}
confidence_vals = []
for block_num in blocks:
    stim_start_end[block_num] = {}
    # get stim order file
    stim_order_fnm = os.path.join(eeg_dir, choose_hc_sp, 
                                subj_num, "original",
                                "_".join([block_num, "stimorder.mat"])
                                )
    block_stim_order = spio.loadmat(stim_order_fnm, squeeze_me=True)['StimOrder']
    # get recording envelope
    rec_wav = subj_eeg[block_num][:,-1]
    rec_env = np.abs(signal.hilbert(rec_wav))
    if segment_blocks:
        lp_rec_env = utils.get_lp_env(rec_env, lpf_cut, fs_eeg)
        _, block_on_off = utils.set_thresh(rec_wav, lp_rec_env, fs_eeg)
        if which_corr.lower() == 'envs':
            block_segments = utils.segment(rec_env, block_on_off)
        elif which_corr.lower() =='wavs':
            block_segments = utils.segment(rec_wav, block_on_off)
        else:
            raise NotImplementedError(f"which_corr = {which_corr} is not an option")
    else:
        if which_corr.lower() == 'envs':
            x = rec_env
        elif which_corr.lower() =='wavs':
            x = rec_wav
        else:
            raise NotImplementedError(f"which_corr = {which_corr} is not an option")
    prev_end=0 #TODO: verify this doesn't introduce new error (check w finished subject?)
    for stim_ii, stim_nm in enumerate(block_stim_order):
        print(f"processing stim {stim_ii+1} of {block_stim_order.size}")
        # grab stim wav
        stim_wav_og = utils.get_stim_wav(stims_dict, stim_nm, has_wav=True)
        stim_dur = (stim_wav_og.size - 1)/fs_audio
        #TODO: what if window only gets part of stim? 
        # downsample current stim and get envelope  
        sos = signal.butter(8, fs_eeg/3, fs=fs_audio, output='sos')
        stim_wav = signal.sosfiltfilt(sos, stim_wav_og)
        # will "undersample" since sampling frequencies not perfect ratio
        stim_wav = signal.resample(stim_wav, int(np.floor(stim_dur*fs_eeg))) 
        stim_env = np.abs(signal.hilbert(stim_wav))
        if which_corr.lower() == 'wavs':
            y = stim_wav
        elif which_corr.lower() == 'envs':
            y = stim_env
        # prev_sync = int(0)
        if segment_blocks:
            for segment_ii, x in enumerate(block_segments):
                if (x.size < stim_wav.size) and not (segment_ii+1) == len(block_segments):
                    # try next segment
                    continue
                elif (x.size < stim_wav.size) and (segment_ii+1) == len(block_segments):
                    # stim missing and no confidence will be calculated
                    stim_start_end[block_num][stim_nm] = (None, None, None)

                print("checkpoint: matching waves")
                # NOTE: stim_on_off is relative to segment start, not full block recording
                stim_on_off, confidence_val = utils.match_waves(x, y, confidence_lims, fs_eeg)
                print("checkpoint: waves matched")
                if stim_on_off is not None:
                    # get start,end index for current stim relative to block start:
                    stim_start = np.where(stim_on_off)[0][0] + np.where(block_on_off)[0][0]
                    stim_end = np.where(stim_on_off)[0][1] + np.where(block_on_off)[0][0]
                    # save indices and confidence 
                    stim_start_end[block_num][stim_nm] = (stim_start, stim_end, confidence_val)
                    # don't look at recording up to this point again
                    block_segments[segment_ii] = x[np.where(stim_on_off)[0][0]:] 
                    # move onto next stim
                    break
                else:
                    # missing stims
                    stim_start_end[block_num][stim_nm] = (None, None, confidence_val)
                    # move onto next stim
                    break
        else:
            if (x.size < stim_wav.size):
                stim_start_end[block_num][stim_nm] = (None, None, None)
                continue

            print("checkpoint: matching waves")
            stim_on_off, confidence_val = utils.match_waves(x, y, confidence_lims, fs_eeg)
            print("checkpoint: waves matched")
            if stim_on_off is not None:
                # if stim_ii == 0:
                #     prev_end=0#TODO: need to set this somehow if first stim not found... 
                # get start,end index for current stim relative to previous stim end:
                stim_start = np.where(stim_on_off)[0][0] + prev_end
                stim_end = np.where(stim_on_off)[0][1] + prev_end
                # save indices and confidence 
                stim_start_end[block_num][stim_nm] = (stim_start, stim_end, confidence_val)
                # don't look at recording up to this point again
                if which_corr.lower() == 'envs':
                    x = rec_env[stim_end:]
                elif which_corr.lower() =='wavs':
                    x = rec_wav[stim_end:]
                # mark end time of last stim found
                # NOTE: not sure if this will work if first stim in block can't be found
                prev_end = stim_end 
                # move onto next stim
                continue
            else:
                # missing stims
                stim_start_end[block_num][stim_nm] = (None, None, confidence_val)
                # move onto next stim
                continue
#%%
#  save stim timestamps
with open(subj_num+'_stim_start_end.pkl', 'wb') as f:
    pickle.dump(stim_start_end, f)
        
#TODO: all the stuff below here should be on a separate py script so we can update stuff without loosing segmented data
#%% 
# slice eeg data and get stim/responses on single data frame and save df
# NOTE: might run out of memory saving all this eeg data
subj_data = utils.align_responses(subj_eeg, stim_start_end, stims_dict)
#NOTE: Subj 3316 has all stim wavs in the saved pkl which is undesireable
subj_data.to_pickle(subj_num+'_subj_data.pkl')
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
# downsample stims and get envelopes for trf analysis
#TODO: make sure we have way to keep downsampling consistent here as in xcorr bit
stim_envs = []
for ii, stim_nm in enumerate(subj_data['stim_nms']):
    if np.isnan(subj_data['eeg'].iloc[ii]).any():
        stim_envs.append(None)
    else:
        stim_wav = utils.get_stim_wav(stims_dict, stim_nm, has_wav=True, noisy_or_clean='clean')
        stim_dur = (stim_wav.size - 1)/fs_audio
        # downsample current stim and get envelope  
        sos = signal.butter(8, fs_eeg/3, fs=fs_audio, output='sos')
        stim_wav = signal.sosfiltfilt(sos, stim_wav)
        # will "undersample" since sampling frequencies not perfect ratio
        stim_wav = signal.resample(stim_wav, int(np.floor(stim_dur*fs_eeg))) 
        stim_envs.append(np.abs(signal.hilbert(stim_wav)))
subj_data['stim_envs'] = stim_envs
#%% 
# compute backward trf, cross-validate
from mtrf.model import TRF
from mtrf.stats import cross_validate
bwd_trf = TRF(direction=-1)
tmin = 0.0
tmax = 0.4
lam= np.power(10.0, np.arange(-5,5)) # powers of 10
n_test = 50 #number of trials to withold for testing
#TODO: probably want this to be more randomized manner, especially since subj_data sorted by confidence vals 
envs_train, eeg_train = subj_data['stim_envs'].dropna().tolist()[:-n_test], subj_data['eeg'].dropna().tolist()[:-n_test]

envs_test, eeg_test = subj_data['stim_envs'].dropna().tolist()[-n_test:], subj_data['eeg'].dropna().tolist()[-n_test:]
r_vals, mse_vals = bwd_trf.train(envs_train, eeg_train, fs_eeg, tmin, tmax, regularization=lam)
#%%
# r_bwd, mws_bwd = cross_validate(bwd_trf, subj_data['stim_envs'].dropna().tolist(), subj_data['eeg'].dropna().tolist())
                                
#%%
# verify that stim and response durs are the same now?
stim_durs = [(s.size - 1)/fs_eeg for s in subj_data.loc[subj_data['eeg'].notna(), 'stim_envs']]
response_durs = [(r.size - 1)/fs_eeg for r in subj_data.loc[subj_data['eeg'].notna(), 'eeg_audio']]
print(np.all(response_durs == stim_durs))
# NOTE: now all the same length!
#%%


#%%
# sort by confidence and visually compare
s_num = 7 # which to plot, in order of confidence
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
sd.play(recording, fs_eeg)
#%%
# NOTE: needa update verification variables below

#%%
# # inspect individual stimulus
# s_num = 100
# fig, ax = plt.subplots(nrows=2)
# t_vecs = [utils.make_time_vector(fs, v.size) for fs, v in zip([fs_audio, fs_eeg], [S_noisy[s_num][0], R_audio[s_num]])]

# ax[0].plot(t_vecs[0], S_noisy[s_num][0])
# ax[1].plot(t_vecs[1], R_audio[s_num])
# [axx.set_title(f'{title}') for axx, title in zip(ax.flatten(), ['OG Stim', 'Audio Recording'])]
# fig.suptitle(f'{S_names[s_num][:-4]}')
# plt.show()
# #TODO: function to look up subj_eeg responses by stim name/order regardless of blocks
# #%% 
# # listen to orig
# sd.play(S_noisy[s_num][0], fs_audio)
# #%%
# # listen to recording
# sd.play(R_audio[s_num], fs_eeg)

# %%
# save results

# with open(subj_num+'_segmented_eeg.pkl', 'wb') as f:
#     pickle.dump(R, f)
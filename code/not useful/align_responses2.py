# align stimuli to eeg recording, more complicated
# %%
import scipy.io as spio
import numpy as np
import os
import sounddevice as sd
import h5py
from scipy import signal
#%% 
# Get stim information file NOTE: not sure which of these is right/if different?
choose_stim_fl = 1 # 1 or 2
stim_fnm1 = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\Matlab_scripts EEG\EEG_Study\master_stim_file_schiz_study_old.mat"
stim_fnm2 = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\Matlab_scripts EEG\EEG_Study\master_stim_file_schiz_studybis.mat"

stim_fls = {1: stim_fnm1, 2: stim_fnm2}

all_stims_mat = spio.loadmat(stim_fls[choose_stim_fl], squeeze_me=True)

# NOTE: I think "orig_clean" and "orig_noisy" are regular wavs and 
#   "aud_clean"/"aud_noisy" are 100-band spectrograms?
# according to master_docEEG spectrograms are at 100 Hz? 
all_stims = all_stims_mat['stim']
# convert structured array to a dict based on dtype names 
# (which correspond to matlab struct fields)
stims_dict = {field_nm : all_stims[field_nm][:] for field_nm in all_stims.dtype.names}

# %% 
# Select Subject
eeg_dir = r"C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Ed - Columbia Sz\EEG\EEG For Emeline\Schizophrenia EEG Study\Subject Data and Results"
choose_hc_sp = "hc" # healthy control ("hc") or schizophrenia patients ("sp")
subj_num = '3253' # arbitrarily choose one that has all eeg hdf5s and stim mats
n_blocks = 6
blocks = [f"B{ii}" for ii in range(1, n_blocks+1)]

fs_audio = stims_dict['fs'][0] # 11025 originally
fs_eeg = 2400

eeg_fnms = [fnm for fnm in os.listdir(os.path.join(eeg_dir, choose_hc_sp, subj_num, "original")) if fnm.endswith('.hdf5')]
#NOTE: below assumes data for all six blocks present in original folder
eeg_fnms_dict = {block_num: fnm for block_num, fnm in zip(blocks, eeg_fnms)}
subj_eeg = {}
subj_times = {} #start stop times     
# %% 
# sync stimuli

for block_num in blocks:
    # step size (in seconds) for xcorr calculation window
    #NOTE: "NeuralFindEvent_Schizophrenia" uses these step sizes but "NeuralFindEventNoisy"
    # used a 20s window for all stims? went with these because seems 20s is too large for the stimuli, 
    # which are about a second long, although not sure why they'd be shorter in block 6?
    if int(block_num[-1]) == 6:
        step = 1
    else:
        step = 5
    # get eeg raw data
    eeg_fnm = os.path.join(eeg_dir, choose_hc_sp, subj_num, "original",
                           eeg_fnms_dict[block_num])
    block_file = h5py.File(eeg_fnm) #returns a file; read mode is default
    subj_eeg[block_num] =  np.asarray(block_file['RawData']['Samples'])
    del block_file
    # get stim order file
    stim_order_fnm = os.path.join(eeg_dir, choose_hc_sp, 
                                subj_num, "original",
                                "_".join([block_num, "stimorder.mat"])
                                )
    block_stim_order = spio.loadmat(stim_order_fnm, squeeze_me=True)['StimOrder']
    # loop thru stimuli in current block
    start_ii = 0 # a very good place to start
    subj_times[block_num] = {} # init nested dict
    # audio on last eeg channel
    recorded_audio = subj_eeg[block_num][:,-1]
    for wav_nm in block_stim_order:
        idx = stims_dict['ID'] == wav_nm[:-4]
    # find wav for current stimulus
        noisy_wvform = stims_dict['orig_noisy'][idx][0]
        # print(f"noisy_wvform.size before: {noisy_wvform.size}")
        wav_dur = (fs_audio)**-1 * (noisy_wvform.size - 1) # duration in s
        sos = signal.butter(8, fs_eeg/3, fs=fs_audio, output='sos')
        noisy_wvform = signal.sosfiltfilt(sos, noisy_wvform) # antialias filt
        
        # downsample
        noisy_wvform = signal.resample(noisy_wvform, 
                                           int(np.floor(wav_dur*fs_eeg)))
            # print(f"noisy_wvform.size after: {noisy_wvform.size}")
            # raise NotImplementedError
        #TODO: figure out why xcorr aray size isn't what we're hoping for
        xcorr = signal.correlate(recorded_audio[start_ii:], noisy_wvform, 
                                     mode="same")
        # normalize by waveform energy
        #NOTE: didn't really make values much smaller?
        xcorr /= (noisy_wvform**2).sum()
        # except:
            # print(block_num, start_ii, recorded_audio.size, noisy_wvform.size)
        # correlate noisy waveform w recording to find start "lag"
        #NOTE: correlate in "valid" mode gives the start, not "same" as I'd thought...
        #NOTE: they took the max of absolute correlation in matlab script...
        # I don't think abs val is needed?
        stim_start = np.argmax(np.abs(xcorr))

        stim_end = stim_start + noisy_wvform.size
        # start search on next stimulus ignoring previous
        # start_ii += stim_end
        subj_times[block_num][wav_nm[:-4]] = (stim_start, stim_end)
    del block_stim_order
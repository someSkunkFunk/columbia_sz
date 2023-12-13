#%%
# import packages
import pickle
import scipy.io as spio
import numpy as np
import os
# import sounddevice as sd
import h5py
from scipy import signal
from scipy.stats import pearsonr
import utils
from mtrf.model import TRF
from mtrf.stats import permutation_distribution
#%%
#  TRF formatting helper functions
#NOTE: might be good to have the data in TRF (stimulus,response) format already so don't have to do all the setup each time?

# downsample audio and get envelopes #NOTE: should only do once not on bluehive so doesn't take forever to run
def get_stim_envs(stims_dict, clean_or_noisy, fs_output):
    ''''
    stims_dict: classic stims dict with both dirty and clean stims from stim file
    clean_or_noisy: choose which stims to downsample and extract envelopes from "clean" or "noisy"
    fs_output: desired sampling frequency for resulting envelopes
    returns: stim envelopes sampled at fs_output
    '''
    fs_stim = stims_dict['fs'][0]
    sos = signal.butter(3, fs_output/3, fs = fs_stim, output='sos')
    # get envelopes first, then resample
    stim_envs = {stim_nm: np.abs(signal.hilbert(stim_wav)) 
                 for stim_nm, stim_wav in zip(stims_dict['ID'], stims_dict["orig_"+clean_or_noisy])}


    ds_envs = {stim_nm: signal.resample(signal.sosfiltfilt(sos, stim_wav), int((stim_wav.size - 1)*fs_output/fs_stim)) 
                for stim_nm, stim_wav in stim_envs.items()}
    
    return ds_envs

# set up stimuli and responses for trf
def get_time_between_stims(stim_start_end, fs):
    times_between = {}
    for block in stim_start_end.keys():
        times_between[block] = {} 
        #NOTE: I believe start/end are sample indices but should double check
        prev_end = None
        prev_stim_nm = None
        for stim_nm, (start, end, _) in stim_start_end[block].items():
            if start is not None:
                if prev_end is None:
                    # first stim
                    prev_end = end
                    prev_stim_nm = stim_nm
                else:
                    # record time difference in samples and time
                    trans_nm = prev_stim_nm+stim_nm
                    times_between[block][trans_nm] = ( int(start-prev_end), (start - prev_end)/fs)
                    prev_end = end
                    prev_stim_nm = stim_nm
            else:
                # skip missing stims
                continue 
    return times_between

def setup_xy(subj_data, stim_envs, reduce_trials_by=None, outlier_idx=None):
    stimulus = []
    response = []
    prev_nm = None
    prev_block = None
     
    # need timestamps to reduce by pauses
    if reduce_trials_by == "pauses":
            # stitch trials together by block/pauses rather than stories;
            #  should give more than 18 trial
            fs_og = 2400 #NOTE: timestamps fs at og eeg fs but that info not in pickle file
            timestamps_fnm = os.path.join(os.getcwd(), "..", 'eeg_data', subj_cat, subj_num, subj_num+"_stim_start_end.pkl")
            with open(timestamps_fnm, 'rb') as f:
                stim_start_end = pickle.load(f)
            times_between = get_time_between_stims(stim_start_end, fs_og)
    
    for stim_nm in subj_data.dropna()['stim_nms']:
        s = np.squeeze(stim_envs[stim_nm[:-4]])
        r = np.asarray(subj_data['eeg'].loc[subj_data['stim_nms']==stim_nm])[0]
        # filter out bad electrodes when given mask for outliers
        if outlier_idx is not None:
            r = r[:,~outlier_idx]
        # check that s, r lengths match, if not drop extra sample
        
        if abs(s.shape[0]- r.shape[0]) == 1:
            # sometimes they're off by one sample
            #NOTE: maybe check if the stimuli that are off by one sample correspond with those that have overlapping timestamps
            end = min([s.shape[0], r.shape[0]])
            # trim extra sample
            s, r = s[:end], r[:end,:]   
        if reduce_trials_by is not None:
            if reduce_trials_by == "stim_nm":
                if prev_nm is not None:
                    if stim_nm[:6] == prev_nm[:6]:
                        # concatenate to prev stim if in same block and story
                        stimulus[-1] = np.concatenate([stimulus[-1], s])
                        response[-1] = np.concatenate([response[-1], r])
                        prev_nm = stim_nm
                    else:
                        # not same story, make new list element
                        stimulus.append(s)
                        response.append(r)
                        prev_nm = stim_nm
                else:
                    # first stim
                    stimulus.append(s)
                    response.append(r)
                    prev_nm = stim_nm
            elif reduce_trials_by == "pauses":
                block = stim_nm[:3:2].capitalize()
                # stim_ps = times_between[block]
                if prev_nm is not None and prev_block == block:
                    # get number of samples between previous and current stim
                    pause_tm, _ = times_between[block][prev_nm+stim_nm]
                    if pause_tm <= 1:
                        # concatenate to prev stim if in same block and story
                        stimulus[-1] = np.concatenate([stimulus[-1], s])
                        response[-1] = np.concatenate([response[-1], r])
                        prev_nm = stim_nm
                    else:
                        # puase too long, make new list element
                        stimulus.append(s)
                        response.append(r)
                        prev_nm = stim_nm
                else:
                    # first stim (or frist block)
                    stimulus.append(s)
                    response.append(r)
                    prev_nm = stim_nm
                prev_block = block
                    
                
        
        else:
            # Don't reduce trials at all
            stimulus.append(s)
            response.append(r)
    return stimulus, response
#%%
# select data
subj_cat="hc"
subj_num="3316"
noisy_or_clean = "clean"
trial_sep = "pauses" #how "trials" were separated
fnm = os.path.join(os.getcwd(),"..", 
                   "results",subj_cat, subj_num, 
                   "_".join([subj_num, noisy_or_clean, 
                             trial_sep,"env_recon_results.pkl"]))
with open(fnm, 'rb') as file:
    results = pickle.load(file)
# load original stims
# TODO: replace w downsampled stims before running on bluehive?
stims_fnm = os.path.join(os.getcwd(), "..", "eeg_data", "stims_dict.pkl")

with open(stims_fnm, 'rb') as file:
    stims_dict = pickle.load(file)


# get individual subject data
eeg_dir = os.path.join(os.getcwd(), "..", 'eeg_data')
subj_num = '3253' # TODO: loop thru multiple subjects reading in from sbatch script
subj_cat = "hc"
subj_data_fnm = subj_num+"_prepr_data.pkl" # pandas dataframe structure
with open(os.path.join(eeg_dir, subj_cat, subj_num, subj_data_fnm), 'rb') as file:
    subj_data = pickle.load(file)
fs_eeg = subj_data['fs'][0]
#%%
# set up TRF stuff
lam = np.median(results['best_lam'])
r_obs = np.mean(results['r_ncv'])
tmin, tmax = 0, 0.4  # range of time lags
trf = TRF(direction=-1)
stim_envs = get_stim_envs(stims_dict, noisy_or_clean, fs_output=fs_eeg)
stimulus, response = setup_xy(subj_data, stim_envs, trial_sep, None)
#%%
# significance test

r_perm = permutation_distribution(trf, stimulus, response, fs_eeg, tmin, tmax, lam, n_permute=1000,k=5)
p = sum(r_perm>=r_obs)/r_perm.size
import matplotlib.pyplot as plt
plt.hist(r_perm, bins=100)
plt.axvline(x=r_obs, ymin=0, ymax=1, color='black',linestyle='--')
plt.annotate(f'p={p.round(2)}')
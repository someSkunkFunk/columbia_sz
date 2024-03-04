# script for mtrf analysis on Ed's test dataset to make sure problems in my trf code aren't the problem
#%%
# import packages
import pickle
import scipy.io as spio
import numpy as np
import os
import re

import utils
from trf_helpers import find_bad_electrodes, get_stim_envs, setup_xy

from mtrf.model import TRF
from mtrf.stats import crossval, nested_crossval


    
#%%

    
data_loc=os.path.join("..","Ed")
eeg_dir=os.path.join(data_loc,r"EEG Data Hong Kong")
envs_dir=os.path.join(data_loc,"Envelopes")
envs={}
eeg={}
pattern = r'envelope(\d{1,2})_128Hz\.mat'
for fnm in os.listdir(envs_dir):
    fl_pth=os.path.join(envs_dir,fnm)
    match=re.match(pattern,fnm)
    if match:
        n=int(match.group(1))
        if 1<=n<=20:
            envs[fnm.strip('.mat')]=spio.loadmat(fl_pth)['envelope']
# sort envelopes and trim to assemble stimulus list
pattern=r'envelope(\d{1,2})_128Hz'
sorted_stims=[key for key in envs.keys()]
sorted_stims.sort(key=lambda x: int(re.match(pattern, x).group(1)))
stimulus=[envs[s][:19840] for s in sorted_stims]
eeg_fnms_sorted=os.listdir(eeg_dir)
eeg_fnms_sorted.sort()
for fnm in eeg_fnms_sorted:
    fl_pth=os.path.join(eeg_dir,fnm)
    # needs to be list to avoid mtrfpy error
    eeg[fnm[:2]]=[r for r in spio.loadmat(fl_pth)['resp'][0]]


#%%
# TRF analysis
trf_mode="crossval"
print(f"initializing TRF using {trf_mode}")

fs=128#Hz
tmin=0.0
tmax=0.4
k=5 #NOTE: ed probably did loo
trf=TRF(direction=-1)
for subj in eeg:
    response=[r for r in eeg[subj]]
    if trf_mode=="nested_cv":
        reg=np.logspace(-1, 8, 10)
        r_ncv, best_lam=nested_crossval(trf, stimulus, response, fs, tmin, tmax, reg, k)
        print(f"{subj} r-values: {r_ncv}, mean: {r_ncv.mean().round(3)}")
    elif trf_mode=="crossval":
        reg=100 #to match Ed's numbers
        # _=trf.train(stimulus, response, fs, tmin, tmax, reg, k)
        r_bkwd=crossval(trf, stimulus, response, fs, tmin, tmax, reg)
        print(f"{subj} r-value: {r_bkwd.round(3)}")


                     
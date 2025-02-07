# surprisal-based linear mixed effects model of envelope reconstruction accuracy per word
# note: I recall finding some issue with rms envelope (I think it was a lack of sufficient antialiasing filter) 
# calculation which made it potentially problematic to use that algorithm with other project, so might want to verify 
# envelope reconstruction analyses with more efficacious envelope representations 

#TODO: figure out if we updated the envelopes with the better ones, per note above?
# RE: seems not, based on reading utils.load_matlab_envs code... TODO: replace with the new envelopes (will probably have to rerun env reconstructions)
#%%
import pickle as pkl
import utils
import numpy as np
subj_trf_flpth='../results/evnt_decimate/thresh_000_5fold_shuffled_b01b02b03b04b05/sp/3244/bkwd_trf_noisy_rms_stims_nested.pkl'
with open(subj_trf_flpth, 'rb') as fl:
    trf=pkl.load(fl)


#%%
# checking what is contained in trfs that we saved because I forgot:
# for model in trf['trained_models']:
#     print(np.all(model.weights==trf['trf_fitted'].weights))
# seems like all the models have the same weights - this indicates they are not actually from 
# different data partitions in nested crossvalidation procedure
# ignoring this for now to continue with rest of the code
#TODO: choose optimal lambda value, train on entire dataset with that value, predict envelopes, 
# split (reconstructed AND true) envelopes along word onsets/offsets, pair with surprisal values, run lme model
stim_envs=utils.load_matlab_envs('clean')
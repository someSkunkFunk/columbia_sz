# surprisal-based linear mixed effects model of envelope reconstruction accuracy per word
# note: I recall finding some issue with rms envelope (I think it was a lack of sufficient antialiasing filter) 
# calculation which made it potentially problematic to use that algorithm with other project, so might want to verify 
# envelope reconstruction analyses with more efficacious envelope representations 
#%%
import pickle
flpth='../results/evnt_decimate/thresh_000_5fold_shuffled_b01b02b03b04b05/sp/3244/bkwd_trf_noisy_rms_stims_nested.pkl'
with open(flpth, 'rb') as fl:
    trf=pickle.load(fl)

model=trf['trained_models'][0]
#TODO: choose optimal lambda value, train on entire dataset with that value, predict envelopes, 
# split (reconstructed AND true) envelopes along word onsets/offsets, pair with surprisal values, run lme model
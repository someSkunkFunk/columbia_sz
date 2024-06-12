# plotting script for lambda curve
#%%
# INIT
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils

subjs_list=utils.get_all_subj_nums()
#NOTE THRESH FOLDS DIR HARDCODED BUT THIS MAY BE CUMBERSOME IN FUTURE
lambda_dir=os.path.join("..","results","evnt_decimate",
"thresh_750_5fold_shuffled")
noisy_or_clean="clean"
rms_str="_rms"
lambda_fnm=f"lambda_tuning_curve_{noisy_or_clean}{rms_str}.pkl"
for subj_num in subjs_list:
    subj_cat=utils.get_subj_cat(subj_num)
    figs_dir=os.path.join("..","figures",
    "lambda_tuning_curve",subj_cat,subj_num)
    if not os.path.isdir(figs_dir):
        os.makedirs(figs_dir,exist_ok=True)
    subj_lambda_pth=os.path.join(lambda_dir,subj_cat,subj_num,lambda_fnm)
    with open(subj_lambda_pth, 'rb') as fl:
        subj_lambda=pickle.load(fl)
    print(f"{subj_num} loaded")
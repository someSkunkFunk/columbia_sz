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
noisy_or_clean="noisy"
rms_str="_rms"
lambda_fnm=f"lambda_tuning_curve_{noisy_or_clean}{rms_str}.pkl"
def extract_tuning_curve(subj_lambda:list):
    if subj_lambda[0]!=('regularization', 'mean_r', 'trf_model'):
        raise NotImplementedError("Data seems not to be formatted properly.")
    lambdas=np.zeros(len(subj_lambda[1:]))
    r_vals=np.zeros_like(lambdas)
    for indx, data in enumerate(subj_lambda[1:]):
        lambdas[indx]=data[0]
        r_vals[indx]=data[1]

    return lambdas, r_vals

#%%
# EXEC
for indx,subj_num in enumerate(subjs_list):
    #load data
    print(f"loading subj {indx+1} of {len(subjs_list)}...")
    subj_cat=utils.get_subj_cat(subj_num)
    figs_dir=os.path.join("..","figures",
    "lambda_tuning_curve",subj_cat,subj_num)
    if not os.path.isdir(figs_dir):
        os.makedirs(figs_dir,exist_ok=True)
    subj_lambda_pth=os.path.join(lambda_dir,subj_cat,subj_num,lambda_fnm)
    with open(subj_lambda_pth, 'rb') as fl:
        subj_lambda=pickle.load(fl)
    print(f"{subj_num} loaded")
    #NOTE: assuming all the same lambdas
    if indx==0:
        lambdas,r_vals=extract_tuning_curve(subj_lambda)
    else:
        _,r_vals=extract_tuning_curve(subj_lambda)
    # plot data
    fig,ax=plt.subplots()
    ax.plot(lambdas,r_vals)
    ax.set_title(f"{subj_num} tuning curve")
    ax.set_xlabel("lambda")
    ax.set_ylabel("r-value")
    # utils.rm_old_figs(figs_dir) #NOTE DECIDED AGAINST RM_OLD_FIGS since both noisy and clean will be put in same folder
    fig_nm=f"{noisy_or_clean}_tuning curve.png"
    fig_pth=os.path.join(figs_dir,fig_nm)
    print(f"Saving fig to path: {fig_pth}")
    plt.savefig(fig_pth)
    plt.close()
    del r_vals, subj_lambda
print("Finished plotting all subjects!")

# %%

# %%
# INIT
import pickle
import numpy as np
import os
import utils
from scipy.stats import pearsonr

#%%
#EXEC
do_plots=True
corrs_b15={}
corrs_b6={}
which_envs_str="_rms_"
noisy_envs=utils.load_matlab_envs('noisy')
# note: there must be a way to bypass having to reload the stims directory every time we call this function within the same script?
clean_envs=utils.load_matlab_envs('clean')

for nm in clean_envs:
        if nm.startswith('b06'):
            corrs_b6[nm]=pearsonr(clean_envs[nm],noisy_envs[nm]).statistic
        else:
            corrs_b15[nm]=pearsonr(clean_envs[nm],noisy_envs[nm]).statistic

# separate block 6 from rest of data since they should be perfectly correlated there

#%%
#plot the values
if do_plots:
    import matplotlib.pyplot as plt
    for blocks,results in zip(["blocks 1-5", "block 6"],[corrs_b15, corrs_b6]):
        results_arr=np.array([r for _,r in results.items()])
        fig,ax=plt.subplots()
        ax.boxplot(results_arr)
        ax.set_title(f"{blocks} noisy-clean envelope correlations")
        if blocks=="block 6":
            blocks_str="b6"
        else:
            blocks_str="b15"
        fig_nm=f"{blocks_str}{which_envs_str}correlations.png"
        fig_pth=os.path.join("..","figures",fig_nm)
        plt.savefig(fig_pth)
        plt.show()


# save results
results_pth=os.path.join("..","results",f"{which_envs_str}_envs_correlation.pkl")
with open(results_pth, 'wb') as file:
    pickle.dump(results,file)



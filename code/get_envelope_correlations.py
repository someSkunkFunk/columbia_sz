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
results={}
which_envs_str="_rms_"
noisy_envs=utils.load_matlab_envs('noisy')
# note: there must be a way to bypass having to reload the stims directory every time we call this function within the same script?
clean_envs=utils.load_matlab_envs('clean')
for nm in clean_envs:
        results[nm]=pearsonr(clean_envs[nm],noisy_envs[nm]).statistic

#plot some the values
if do_plots:
    import matplotlib.pyplot as plt
    results_arr=np.array([r for _,r in results.items()])
    fig,ax=plt.subplots()
    ax.boxplot(results_arr)
    ax.set_title(f"correlation values for noisy vs clean {which_envs_str}")
    fig_nm=f"{which_envs_str}_correlations.png"
    fig_pth=os.path.join("..","figures",fig_nm)
    plt.savefig(fig_pth)
    plt.show()


# save results
results_pth=os.path.join("..","results",f"{which_envs_str}_envs_correlation.pkl")
with open(results_pth, 'wb') as file:
    pickle.dump(results,file)



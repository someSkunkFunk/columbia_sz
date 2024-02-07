#%%
import utils
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
#%% get segmented data
all_subjs=utils.get_all_subj_nums()
if __name__=='__main__':
    data_dir=os.path.join("..","eeg_data","preprocessed_xcorr")
    ii=0
    subj_num=all_subjs[ii]
    subj_cat=utils.get_subj_cat(subj_num)
    subj_data=utils.load_preprocessed(subj_num,evnt=False,which_xcorr='envs')

    fig,ax=plt.subplots()
    ax.hist
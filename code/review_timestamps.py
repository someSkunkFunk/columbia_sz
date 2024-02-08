#%%
# setup
import utils
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

all_subjs=utils.get_all_subj_nums()
which_xcorr="wavs" # clean wavs seem to yield best results....
stims_dict=utils.get_stims_dict()
all_stim_nms=stims_dict["ID"] #note .wav not in name
#%%
#NOTE: not really using the confidence vals for anything but maybe will in the future 
# so leaving some seemingly useless code for it in
if __name__=='__main__':
    # subj_data=utils.load_preprocessed(subj_num,evnt=False,which_xcorr='envs')
    found_stims_mat=np.zeros((len(all_subjs),len(all_stim_nms)))

    for subj_ii,subj_num in enumerate(all_subjs):
        print(f"{subj_num}")
        subj_cat=utils.get_subj_cat(subj_num)
        timestamps_pth = os.path.join("..","eeg_data","timestamps",subj_cat,subj_num,
                                  f"{which_xcorr}_timestamps.pkl")
        with open(timestamps_pth, 'rb') as f:
            timestamps=pickle.load(f)
        #flatten stim names
        temp_subj_nms_flt=[]
        temp_conf_vals_flt=[]
        temp_found_stims=[]
        for block in timestamps:
            for stim_nm, (start,end,confidence) in timestamps[block].items():
                temp_subj_nms_flt.append(stim_nm.strip(".wav"))
                temp_conf_vals_flt.append(confidence)
                if all([start,end,confidence]):
                    temp_found_stims.append(1)
                else:
                    # using -1 as false since i want nans to be 0
                    temp_found_stims.append(-1)
        if any([f==1 for f in temp_found_stims]):
            print(f"found {sum([f==1 for f in temp_found_stims])} out of {len(temp_subj_nms_flt)}")
        # put in order and fill in missing vals
        subj_nms_flt=[None for _ in all_stim_nms]
        conf_vals_flt=[None for _ in all_stim_nms]
        found_stims=[None for _ in all_stim_nms]
        for stim_ii,stim_nm in enumerate(all_stim_nms):
            for nm,conf,found in zip(temp_subj_nms_flt,temp_conf_vals_flt,temp_found_stims):
                subj_nms_flt[stim_ii]=nm
                conf_vals_flt[stim_ii]=conf
                found_stims[stim_ii]=found
                found_stims_mat[subj_ii,stim_ii]=found
    # set nans to zero (meaning stim was not even presented)
    found_stims_mat[np.isnan(found_stims_mat)]=0
    plt.imshow(found_stims_mat[:].T,cmap='hot')
    plt.show()
# script takes best lambda value found in training on clean data, re-trains the model using this lambda value on full clean set, then predicts on noisy data
#%%
# INIT
import pickle
import numpy as np
import os

import utils
from trf_helpers import load_stim_envs, setup_xy, sensible_lengths,get_subj_trf_pth, get_thresh_folds_dir
from mtrf.model import TRF
from mtrf.stats import crossval, nested_crossval

#%%
# EXEC
# get best lambda from existing trf results
# since 
thresh_folds_dir=get_thresh_folds_dir(blocks='6') # gets shuffled, 5-fold, thresh750 results by default

all_subjs_list=utils.get_all_subj_nums()
for subj_num in all_subjs_list:
    block6_subj_trf_pth=get_subj_trf_pth(subj_num,thresh_folds_dir)
    with open(block6_subj_trf_pth,'rb') as file:
        subj_trf_b6=pickle.load(file)
        # choose lambda value with maximum r-value....? (not sure if that's a principled use of nested crossvalidation results)
        


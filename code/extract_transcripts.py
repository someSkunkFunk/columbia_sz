#%%
# INIT
import os
import utils
import numpy as np
from string import punctuation
import pickle

stims_dict=utils.get_stims_dict()
#%%
# EXEC

for sentence,nm in zip(stims_dict['String'],stims_dict['Name']):
    story_nm=
    if isinstance(sentence,str):
        pass
    elif isinstance(sentence,np.ndarray):
        pass
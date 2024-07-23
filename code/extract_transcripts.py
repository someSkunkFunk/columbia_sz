#%%
# INIT
import os
import utils
import numpy as np
from string import punctuation, digits
import pickle

stims_dict=utils.get_stims_dict()
#%%
# EXEC
story_nms=[n.replace('./Sounds/','') for n in stims_dict["Name"]]
story_nms=[n.replace('_16K_NM.wav','') for n in story_nms]
story_nms=[n.replace('./Sounds/','') for n in stims_dict["Name"]]
story_nms=[n.replace('_16K_NM.wav','') for n in story_nms]
# generate 
stories=set([n.rstrip(digits) for n in story_nms])
ids=[x for x in stims_dict['ID']]
story_stim_ids=[[nm,x] for nm,x in zip(story_nms,ids)]
for sentence,nm in zip(stims_dict['String'],story_nms):
    
    if isinstance(sentence,str):
        pass
    elif isinstance(sentence,np.ndarray):
        pass

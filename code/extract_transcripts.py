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
# generate set of unique stories since some are repeated with different speaker/noise but doesn't affect surprisal value:
gender_stories=set([n.rstrip(digits) for n in story_nms])
stories=set([nm.split('_')[1] for nm in gender_stories])
exclude_stories={'common'} # since that one is weird on several accounts... although others not really in the clear...
ids=[x for x in stims_dict['ID']]
story_stim_ids=[[nm,x] for nm,x in zip(story_nms,ids)]
# get number of parts for each story, also ensure the number is consistent across blocks:

block_parts_tally={f'b0{ii}':dict.fromkeys(stories) for ii in range(1,7)}
for [nm, stim_id] in story_stim_ids:
    block=stim_id[:3]
    # figure out how to use map function to figure out which story name matches nm..


# TODO: finish stuff below
def check_numbers_consistent(block_parts_tally:dict,exclude_stories:set):
    # check that when story appears in multiple blocks, same number of parts in each block
    pass
    

    




for sentence,nm in zip(stims_dict['String'],story_nms):
    
    if isinstance(sentence,str):
        pass
    elif isinstance(sentence,np.ndarray):
        pass

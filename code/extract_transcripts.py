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
def check_numbers_consistent(block_parts_tally:dict,exclude_stories=None):
    # check that when story appears in multiple blocks, same number of parts in each block
    blocks=list(block_parts_tally.keys())
    # put
    story_counts={k: [None]*len(blocks) for k in block_parts_tally[blocks[0]]}
    for ii,block in enumerate(blocks):
        for story in story_counts:
            story_counts[story][ii]=block_parts_tally[block][story]
        # after collecting all the counts into reduced dict, make sure each block contains the same number 
        # of story parts for each story (or zero)
    bool_shit=[]
    for story,counts in story_counts.items():
        unique_counts=set(counts)
        if unique_counts=={max(counts),0}:
            print(f'{story} is consistent across blocks with {max(counts)} parts :)')
            bool_shit.append(True)
        else:
            print(f'{story} is NOT consistent across blocks with {max(counts)} parts :(')
            bool_shit.append(False)
    return all(bool_shit)
# story_nms_detailed=[n.replace('./Sounds/','').replace('_16K_NM.wav','') for n in stims_dict["Name"]]


# i learned how to use map function finally oooo...
story_nms_detailed=list(map(lambda n: n.replace('./Sounds/','').replace('_16K_NM.wav',''), stims_dict["Name"]))
story_nms=list(map(lambda n: n.split("_")[1].rstrip(digits), story_nms_detailed))
# generate set of unique stories since some are repeated with different speaker/noise but doesn't affect surprisal value:
gender_stories=set([n.rstrip(digits) for n in story_nms_detailed])
stories=set([nm.split('_')[1] for nm in gender_stories])
exclude_stories={'common'} # since that one is weird on several accounts... although others not really in the clear...
ids=[x for x in stims_dict['ID']]
story_stim_ids=[[nm,x] for nm,x in zip(story_nms,ids)]
# get number of parts for each story, also ensure the number is consistent across blocks:

block_parts_tally={f'b0{ii}':dict.fromkeys(stories) for ii in range(1,7)}

for block in block_parts_tally:
    print(f"starting {block}")
    for story in block_parts_tally[block]:
        # assumes speaker prefix has been removed from story_stim_ids:
        block_parts_tally[block][story]=len(list(filter(lambda x: x[1].startswith(block) and x[0]==story,story_stim_ids)))
consistent_bool=check_numbers_consistent(block_parts_tally)
print(f"numbers consistent? {consistent_bool}")
# for [nm, stim_id] in story_stim_ids:

#     block=stim_id[:3]
#     block_parts_tally[block]


# TODO: finish stuff below




    
    

    

    




for sentence,nm in zip(stims_dict['String'],story_nms_detailed):
    
    if isinstance(sentence,str):
        pass
    elif isinstance(sentence,np.ndarray):
        pass

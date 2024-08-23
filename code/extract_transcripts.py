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
def make_story_filter(story_stim_ids,exclude_stories={'hankFour','common'}):
    '''
    makes bool list of stories
    assumes story_stim_ids in presentation order so no need to keep track of blocks
    '''
    keep_stories_filter=[False]*len(story_stim_ids)
    mem={}
    for ii, [story, stim_id] in enumerate(story_stim_ids):
        if story in exclude_stories:
            continue
        # block,part=stim_id[1:3],stim_id[-2:]
        part=stim_id[-2:]
        if not story in mem:
            #init a set
            mem[story]={part}
            keep_stories_filter[ii]=True
        elif story in mem and part not in mem[story]:
            mem[story].add(part)
            keep_stories_filter[ii]=True
    return keep_stories_filter

def group_stories(stims_dict,story_nms,story_filter):
    keep_strings=[(stnc,nm) for boo,stnc,nm in 
    zip(story_filter,stims_dict['String'],story_nms) if boo]
    # keep IDs to link back into correct trials:
    keep_ids=stims_dict['ID'][story_filter]
    
    grouped_strings={'hank':[],'howOne':[],
    'howTwo':[],'howThree':[],'howFour':[]}
    for stnc,nm in keep_strings:
        if isinstance(stnc,np.ndarray):
            stnc=stnc[0] #extract string from array
        # put sentences in proper list
        if nm.startswith('hank'):
            grouped_strings['hank'].append(stnc)
        elif nm.startswith('how'):
            grouped_strings[nm].append(stnc)

    return grouped_strings,keep_ids



story_nms_detailed=utils.get_story_nms(stims_dict,detailed=True)
story_nms=utils.get_story_nms(stims_dict,detailed=False)
# generate set of unique stories since some are repeated with different speaker/noise but doesn't affect surprisal value:
gender_stories=set([n.rstrip(digits) for n in story_nms_detailed])
stories=set([nm.split('_')[1] for nm in gender_stories])
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
story_filter=make_story_filter(story_stim_ids)
for story_info,filter_value in zip(story_stim_ids,story_filter):
    print(story_info,filter_value)
grouped_transcripts,grouped_ids=group_stories(stims_dict,story_nms,story_filter)
print(grouped_transcripts)
#TODO: re-verify that grouped_ids doesn't contain 
print(grouped_ids)
#%%
tr_pth=os.path.join("..","eeg_data","grouped_transcripts.pkl")
ids_pth=os.path.join("..","eeg_data","grouped_ids.pkl")
with open(tr_pth,'wb') as fl:
    pickle.dump(grouped_transcripts,fl)
with open(ids_pth,'wb') as fl:
    pickle.dump(grouped_ids,fl)



# for sentence,nm in zip(stims_dict['String'],story_nms_detailed):
    
#     if isinstance(sentence,str):
#         pass
#     elif isinstance(sentence,np.ndarray):
#         print(sentence)
#         print(sentence[0])
#         print(type(sentence[0]))

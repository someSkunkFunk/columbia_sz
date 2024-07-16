#script for converting textgrid file information into phoneme and word vectors
#%%
# INIT

import os
import utils
import numpy as np

#%%
# EXEC
stim_fl_path=os.path.join("..","eeg_data","stim_info.mat")
stims_dict=utils.get_stims_dict(stim_fl_path)
# strip prefix and suffix in stim names to match textgrid file names

stim_names=[n.replace('./Sounds/','') for n in stims_dict['Name']]
#note pretty sure this means the textgrids have time in 16kHz 
# fs instead of original audio fs
stim_names=[n.replace('_16K_NM.wav','') for n in stim_names]

textgrids_path=os.path.join("..","eeg_data","textgrids")
textgrids_fnms=os.listdir(textgrids_path)

test_stim_nm=stim_names[0]
test_stim_tg_path=os.path.join(textgrids_path,test_stim_nm)
#%%
# assume short format only (and that phone precedes word tier in each case)
def read_textgrid(textgrid_path):
    '''
    '''
    with open(textgrid_path, 'rb') as fl:
        tg=fl.readlines()
    # decode bytes into strings, remove new line chars
    tg=[line.decode('UTF-8')[:-1] for line in tg]
    return tg
def get_word_boundaries(textgrid_path,remove_sp=True):
    _fs=16e3#TODO verify this is correct
    # guessing at 16kHz based on prpsoundobject file names
    tg=read_textgrid(textgrid_path)
    # get start of each textgrid tier
    phn_start=[ii+1 for ii, txt in enumerate(tg[1:-1]) if tg[ii+1]=='"phone"' and tg[ii]=='"IntervalTier"'][0]
    wrd_start=[ii+1 for ii, txt in enumerate(tg[1:-1]) if tg[ii+1]=='"word"' and tg[ii]=='"IntervalTier"'][0]
    # size is 3 after tier class name
    num_phns=int(tg[phn_start+3])
    num_wrds=int(tg[wrd_start+3])
    # extract separate lists for each class
    # N=3
    num_skip=4 # redundant lines before actual phone/word info start
    phns=tg[phn_start+num_skip:wrd_start-1]
    wrds=tg[wrd_start+num_skip:]
    # group into triplets for each item
    N=3
    phns=[phns[n:n+N] for n in range(0,len(phns),N)]
    wrds=[wrds[n:n+N] for n in range(0,len(wrds),N)]
    # sanity check:
    if len(phns)!=num_phns or len(wrds)!=num_wrds:
        raise NotImplementedError('output length doesnt match expected number of words or phones')    
    if remove_sp:
        phns=list(filter(lambda x: x[-1]!='"sp"',phns))
        wrds=list(filter(lambda x: x[-1]!='"sp"',wrds))
    return phns, wrds
tg=read_textgrid(test_stim_tg_path)

phns,wrds=get_word_boundaries(test_stim_tg_path)
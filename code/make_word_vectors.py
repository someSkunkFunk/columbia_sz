#script for converting textgrid file information into phoneme and word vectors
#%%
# INIT

import os
import utils
import numpy as np
import pickle
# assume short format only (and that phone precedes word tier in each case)
def read_textgrid(textgrid_path):
    '''
    '''
    with open(textgrid_path, 'rb') as fl:
        tg=fl.readlines()
    # decode bytes into strings, remove new line chars
    tg=[line.decode('UTF-8')[:-1] for line in tg]
    return tg
def get_word_boundaries(textgrid_path,remove_sp=True,nums2float=True):
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
    if nums2float:
        # convert numbers to float from str
        phns=[[float(x[0]),float(x[1]),x[2]] for x in phns]
        wrds=[[float(x[0]),float(x[1]),x[2]] for x in wrds]
    return phns, wrds


#%%
# EXEC
stim_fl_path=os.path.join("..","eeg_data","stim_info.mat")
stims_dict=utils.get_stims_dict(stim_fl_path)
# strip prefix and suffix in stim names to match textgrid file names

stim_nms=[n.replace('./Sounds/','') for n in stims_dict['Name']]
#note pretty sure this means the textgrids have time in 16kHz 
# fs instead of original audio fs
stim_nms=[n.replace('_16K_NM.wav','') for n in stim_nms]

textgrids_path=os.path.join("..","eeg_data","textgrids")
textgrids_fnms=os.listdir(textgrids_path)
extract_bounds=False
bounds_fl_nm="phn_wrd_bounds.pkl"
bounds_fl_pth=os.path.join("..","eeg_data",bounds_fl_nm)
if extract_bounds:
    boundaries={}
    for ii,stim in enumerate(stim_nms):
        print(f"extracting stim {ii+1} of {len(stim_nms)}...")
        tg_path=os.path.join(textgrids_path,stim)
        phns,wrds=get_word_boundaries(tg_path)
        # save to pkl for later use
        #NOTE this is only going to save bounds for last stim 
        boundaries[stim]={'phones':phns,'words':wrds}
        del phns,wrds
    with open(bounds_fl_pth, 'wb') as fl:
        print("saving boundaries..")
        pickle.dump(boundaries,fl)
    print('saved.')
else:
    print('loading existing bounds from pkl...')
    with open(bounds_fl_pth, 'rb') as fl:
        boundaries=pickle.load(fl)
    print('done.')

# visualize stims along with waveforms to verify fs
def make_bounds_vctrs(bounds:list,fs=16e3,displace_offset=True):
    # shoudl work with either words or phns, but must be fed individually
    # since operating on individual stimulus set of boundaries
    # assumes time in seconds
    print(f"using fs: {fs} to make onset offset vectors") #NOTE UNSURE IF 16kHz or 11025
    # add string prefix and suffix back to enable exact string comparison
    t_max=bounds[-1][1] # assuming last value for phn AND wrds is the same 
    time_vec=np.arange(0,t_max,1/fs)
    on_iis=[None]*len(bounds)
    off_iis=[None]*len(bounds)
    wrds_list=[None]*len(bounds)
    # and corresponds with final word; also units of seconds assumed
    for ii, (on_t, off_t, word) in enumerate(bounds):
        # since boundaries are sparse, just return indices along with time vector, use indices to populate at plotting time
        on_iis[ii]=np.argmin(np.abs(time_vec-on_t))
        if displace_offset:
            # shift offsets so stems don't plot on top of each other for visualization
            _amt_shift=200 # number of samples to shift offset by
            off_iis[ii]=np.argmin(np.abs(time_vec-off_t))-_amt_shift
        wrds_list[ii]=word
    return time_vec,on_iis,off_iis,wrds_list

import matplotlib.pyplot as plt
fs_wavs=stims_dict['fs'][0]
fig_width=20
fig_height=6
show_figs=False
start_from=274 # if restarting due to kernel crash
for ii,(story_nm,bounds) in enumerate(boundaries.items()):
    if ii<start_from:
        continue
    else:
        print(f"plotting {ii+1} of {len(boundaries)}")
        # add the stupid prefix and suffix back to enable exact match str comparison
        full_nm='./Sounds/'+story_nm+'_16K_NM.wav'
        # match the textgrid fl name to appropirate wav
        nm_match_idx=stims_dict['Name']==full_nm 
        #note some are repeated so multiple matches will be returned
        #grab clean wav from first match
        # checked that indexing works with non-repeated stims also
        stim_wav=stims_dict['orig_clean'][nm_match_idx][0]
        wav_id=stims_dict['ID'][nm_match_idx][0]
        t_stim=np.arange(0,stim_wav.size/fs_wavs,1/fs_wavs)
        if t_stim.size>stim_wav.size:
            # sometimes ends up with extra sample, remove it
            t_stim=t_stim[:stim_wav.size]
        bound_t,on_iis,off_iis,wrds_list=make_bounds_vctrs(bounds['words'])
        on_imps=np.zeros(bound_t.shape)
        off_imps=np.zeros(bound_t.shape)
        on_imps[on_iis]+=1
        off_imps[off_iis]+=1

        fig,ax=plt.subplots(figsize=(fig_width,fig_height))
        try:
            ax.plot(t_stim,stim_wav,label=f"ID:{wav_id}")
            ax.stem(bound_t,on_imps,linefmt='green',label='onsets')
            ax.stem(bound_t,off_imps,linefmt='red',label='offsets')
            plt.legend()
            ax.set_title(f'{story_nm}')
            fig_pth=os.path.join("..","figures","word_bounds",f"{story_nm}")
            plt.savefig(fig_pth)
            if show_figs:
                plt.show()
            plt.close()
        except:
            print("something went wrong here")
#%%
# make surprisal values
def load_surprisal():
    surprisal_fl_pth=os.path.join("..","eeg_data","surprisal.bin")
    with open(surprisal_fl_pth,'rb') as fl:
        surprisals=pickle.load(fl)
    return surprisals
def make_surprisal_vector(onsets,surprisal_vals,fs_input,fs_output):
    pass

surprisals=load_surprisal()
 
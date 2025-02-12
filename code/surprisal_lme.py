# NOTE: not all stimuli were usable for surprisal calculation, so we should filter trf results
# also need to account for mismatched sentences somehow
#%%
# INIT
import os
import utils
import numpy as np
from string import punctuation
import pickle as pkl
import matplotlib.pyplot as plt
from mtrf.model import TRF
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
        # and remove quotation marks from strings
        phns=[[float(x[0]),float(x[1]),x[2].replace('"',"")] for x in phns]
        wrds=[[float(x[0]),float(x[1]),x[2].replace('"',"")] for x in wrds]
    return phns, wrds

def make_bounds_vctrs(bounds:list,fs=16e3,displace_offset=True):
    # should work with either words or phns, but must be fed individually
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



# make surprisal values
def load_surprisal():
    surprisal_fl_pth=os.path.join("..","eeg_data","surprisal.bin")
    with open(surprisal_fl_pth,'rb') as fl:
        surprisals=pkl.load(fl)
    return surprisals

def pair_surp_stims(surprisals, stims_dict):
    '''
    ACTUALLY just realized the point of this function is to pair each sentence with surprisal values for that sentence, which is unnecessary since Jin provided surprisals already organized in this fashin now...
    so really all there is to do is verify that the stims words list for each sentence match that of the provided surprisal values, then use those along with textgrid time bounds to make word vectors...
    surprisals: dict - {story_nm: [([word strings], [surprisal numbers])]}

    returns:
    stim_surprisal_dict: {'stim_id':[IDs], 'words': [words]}
    '''
    paired_surprisals={}
    # i think indexing method came from the old surprisals dict having the entire story in one list rather than sentece by sentence....
    # so i muted all lines with srprs_idx
    # look at names just to check that surprisals dict matches info on original stims_dict
    stim_nms=utils.get_story_nms(stims_dict,detailed=False)
    # srprs_idx=0 # keep track of where in surprisals lists we are
    # get current sentence from stims_dict for comparison

    for stim_id,stim_nm,stim_str in zip(stims_dict['ID'],
    stim_nms,
    stims_dict['String']):
        if isinstance(stim_str,str):
            # note: using split without parameter automaticall strips trailing whitespace,
            # which was giving an error before when using split(" ")
            stim_word_list=stim_str.split()
        # for some weird reason some strings return an array??
        elif isinstance(stim_str,np.ndarray):
            stim_word_list=stim_str[0].split()
        else:
            pass
        n_wrds=len(stim_word_list)
        # get words list for current story,sentence:

        # srprs_word_list=surprisals["words"][srprs_idx:srprs_idx+n_wrds]

        #remove punctuation before word-matching
        stim_word_list=[s.translate(str.maketrans('','',punctuation)) for s in stim_word_list]
        srprs_word_list=[s.translate(str.maketrans('','',punctuation)) for s in srprs_word_list]

        #ensure words match in both lists (assumes in same order in surprisals.bin)
        if all([w1.lower()==w2.lower() for w1,w2 in zip(stim_word_list,srprs_word_list)]):
            paired_surprisals['ID']=stim_id
            paired_surprisals['Name']=stim_nm
            
            # paired_surprisals['surprise_values']=surprisals["values"].squeeze()[srprs_idx:srprs_idx+n_wrds]
        else:
            raise NotImplementedError("word lists do not match!")

        #update index
        # srprs_idx+=n_wrds
    return paired_surprisals

def check_wordlists_match(*word_lists):
        '''
        give list of word lists for particular sentence, check that all lists have the same words (for arbitrary number of word lists - at least 2)
        '''
        if not word_lists:
            raise ValueError('no lists.')
        words_match=False
        sentence_len=len(word_lists[0])
        # check all equal
        lengths_match=[sentence_len==len(l) for l in word_lists]
        if all(lengths_match):
            # raise Exception('List lengths need to match!')
            
            for word_indx in range(sentence_len):
                #note: might need to remove spaces and punctuation as well...?
                words_at_indx=[lst[word_indx].lower() for lst in word_lists]
            if len(set(words_at_indx))==1:
                words_match=True
        return words_match

def pair_surprisals_with_boundaries_stupid(surprisals,boundaries,exclude_stories={'hankFour','common'}):
    # this method didn't work for all the stimuli, and I realized was unnecessarily complicated since 
    # we can just keep track of the long names when extracting transcripts for re-mapping
    # get just word boundaries for simplicity
    word_boundaries_temp={lng_nm:stnc['words'] for lng_nm,stnc in boundaries.items()}
    # remove excluded stories
    word_boundaries={}
    for lng_nm,stnc in word_boundaries_temp.items():
        in_exclude=any([xclude_nm in lng_nm for xclude_nm in exclude_stories])
        if in_exclude:
            continue
        else:
            word_boundaries[lng_nm]=stnc
    del word_boundaries_temp

    story_nms=list(surprisals.keys())
    surp_bounds_dict={}
    # keep track of sentence names long for future reference of mismatched stims
    word_bounds_long_nms=list(word_boundaries.keys())
    paired_long_nms=[]
    for stnc_nm_long,stnc_word_bounds in word_boundaries.items():
        # get words for current sentence into separate list
        bound_stnc_list=[x[-1].lower() for x in stnc_word_bounds]
        #TODO: check if punctuation removal/other weird shit needed here
        
        story_nm_mask=[True if short_nm in stnc_nm_long else False for short_nm in surprisals]
        try:
            story_nm_idx=story_nm_mask.index(True)
        except:
            pass
        # get all surprisal organized by sentence from current story
        current_story=story_nms[story_nm_idx]
        current_story_surprisal=surprisals[current_story]
        
        for ii, (surp_stnc_list,stnc_surp_vals) in enumerate(current_story_surprisal): 
            # check that word lists match
            # print(f'{current_story} {ii}')
            words_match=check_wordlists_match(surp_stnc_list,bound_stnc_list)
            if words_match:
                surp_bounds_dict[stnc_nm_long]=(surp_stnc_list,stnc_surp_vals,stnc_word_bounds)
                # remove matched element from surprisals so won't re-check on next sentence
                # surprisals[current_story].pop(ii)
                current_story_surprisal.pop(ii)
                paired_long_nms.append(stnc_nm_long)
                print(f"number remanining in {current_story}: {len(current_story_surprisal)}")
                # print(f"nmber equals number remaining in surprisals[current_story]: {len(surprisals[current_story])==len(current_story_surprisal)}")
                break
                # NOTE: surprisals should ideally have nothing in it by the time this function is done running due to pop?
            
    # print missing long nms before returning paired dict
    all_nms_wbounds, paired_nms=set(word_bounds_long_nms), set(paired_long_nms)
    missing_nms_set=all_nms_wbounds-paired_nms

    # get number of remaining surprisal
    missing_surprisal_stncs=[]
    for story_nm in story_nms:
        sentences_surprisals=surprisals[story_nm]
        for stnc,_ in sentences_surprisals:
            missing_surprisal_stncs.append((stnc,story_nm))
    print(f"{len(missing_surprisal_stncs)} sentences with surprisals not paired with corresponding bounds")
    print(f"sentences with surprisals missing bounds: {missing_surprisal_stncs}")
    

    print(f"{len(missing_nms_set)} out of {len(all_nms_wbounds)} stimuli with bounds not paired.")
    print(f"sentences with bounds missing surprisals:\n{[nm for nm in missing_nms_set]}")

    return surp_bounds_dict
    
# note: seems like below function won't be necessary since surprisals already arranged by sentences
# instead need to match that back to original stimulus ID somehow
# def get_sentence_boundaries(boundaries,story_nm):
#     '''
#     arrange boundaries into sentences for a particular story
#     '''

#     pass
def get_sentence_from_bounds(boundaries,skey):
    '''
    don't need to prefilter just the word bounds
    return sentence as one continuous lowercase string AND list of words
    list useful for comparing with surprisal sentence wordlists
    '''
    sentence_list=[x[-1].lower() for x in boundaries[skey]['words']]
    sentence_str=' '.join(sentence_list)
    return sentence_str, sentence_list

def pair_surprisals_with_boundaries(surprisals,boundaries,surprisal_ids):
    # get just word boundaries for simplicity
    # word_boundaries={lng_nm:stnc['words'] for lng_nm,stnc in boundaries.items()}
    # note word_boundaries will contain all the stimuli not just those used in surprisal transcripts
    paired_surprisal_bounds={}
    mismatched_sentences={}
    for story in surprisals:
        for (surprisal_sentence_list,surprisal_vals),(long_nm,stim_id) in zip(surprisals[story],surprisal_ids[story]):
            bound_sentence_str,bound_sentence_list=get_sentence_from_bounds(boundaries,long_nm)
            words_do_match=check_wordlists_match(bound_sentence_list,surprisal_sentence_list)
            if words_do_match:
                # paired_surprisal_bounds.append((long_nm,boundaries[long_nm],surprisal_vals,stim_id))
                #NOTE: this assumes stim_id is unique... I think that's true...
                paired_surprisal_bounds[stim_id]={'long_name': long_nm,
                'boundaries':boundaries[long_nm], 'surprisal':surprisal_vals
                }
            else:
                print(f"{long_nm} sentences not matched across tg file and stimuli_info file.")
                mismatched_sentences[stim_id]={'long_name':long_nm,             
                    'sentence_words_surprisal': surprisal_sentence_list,
                    'sentence_words_textgrid': bound_sentence_list
                }
                # mismatched_sentences.append((long_nm,surprisal_sentence_list,bound_sentence_list))
    print(f"mismatched_sentences: {mismatched_sentences}")
    return paired_surprisal_bounds, mismatched_sentences
    # still need to loop thru bounds to find which were not paired with a surprisal value.... (those not in exclude stories)

#%%
# EXEC
stim_fl_path=os.path.join("..","eeg_data","stim_info.mat")
stims_dict=utils.get_stims_dict(stim_fl_path)

textgrids_path=os.path.join("..","eeg_data","textgrids")
textgrids_fnms=os.listdir(textgrids_path)


extract_bounds=False
bounds_fl_nm="phn_wrd_bounds.pkl"
bounds_fl_pth=os.path.join("..","eeg_data",bounds_fl_nm)
# NOTE: stuff below doesn't need to run again unless we want to re-read the 
# textgrid files but also if we do then we need to make stim_nms variable for it to run
#  again... which I think came from stims_dict originally...??? 
if extract_bounds:
    # read textgrid info into pkls for easier use - should only need to run once
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
        pkl.dump(boundaries,fl)
    print('saved.')
else:
    print('loading existing bounds from pkl...')
    with open(bounds_fl_pth, 'rb') as fl:
        boundaries=pkl.load(fl)
    print('done.')

# visualize stims along with waveforms to verify fs
make_figs=False
show_figs=False

if make_figs==True:
    fs_wavs=stims_dict['fs'][0]
    fig_width=20
    fig_height=6    
    start_from=0 # if restarting due to kernel crash
    for ii,(story_nm,bounds) in enumerate(boundaries.items()):
        if ii<start_from:
            continue
        else:
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
        
            print(f"plotting {ii+1} of {len(boundaries)}")
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
    
# dict: {'values, 'words'}
surprisals=load_surprisal()
story_nms_detailed=utils.get_story_nms(stims_dict,detailed=True)
# load saved ids for re-mapping surprisals back to sentence nm

ids_pth=os.path.join("..","eeg_data","grouped_ids.pkl")
with open(ids_pth,'rb') as f:
    surprisal_ids=pkl.load(f)
paired_surprisal_bounds,mismatched_sentences=pair_surprisals_with_boundaries(surprisals,boundaries,surprisal_ids)

#
# choose an example subject for first run
subj_trf_flpth='../results/evnt_decimate/thresh_000_5fold_shuffled_b01b02b03b04b05/sp/3244/bkwd_trf_noisy_rms_stims_nested.pkl'
with open(subj_trf_flpth, 'rb') as fl:
    trf=pkl.load(fl)
#%%
# IN DEVELOPMENT: get word-times and compute correlation between stim and reconstruction per word
# then put into a dataframe
def filter_trf_stimuli(trf,surprisal_ids):
    '''
    remove stimuli that don't have surprisal values because reasons
    initiaally thought mismatched sentences should be filtered here too but that would make more sense 
    when computing word-level correlations i think since should be rare and possibly will be able to fix some in future
    '''
    # assume if any of the clips within a trial missing or mismatched, trial is unusable
    #TODO: stim_nms might be confused with other variables here... maybe 
    # change to trial_nms since that's more clear in general?

    # surprisal ids is dict organized by stories, each story has tuples 
    # (both short name, stim_id) - match to stim_id here since that's what trf has
    ids_with_surprisal=[experiment_id for aliases in surprisal_ids.values() for _,experiment_id in aliases]
    # ids_without_bounds=list(mismatched_sentences.keys())
    # NOTE: actually I don't think a trial with a single sentence missing is a problem yet since we 
    # can use word reconstructions accuracies for rest of sentence still if it has surprisal values
    # can filter those on a per-sentence basis when calculating correlationsw
    # NOTE: solution below for filtering assumes relative order in lists being compared doesnt matter
    # keep trials that have surprisal to begin with
    indx_good_trials=[ii for ii, trial_nms in enumerate(trf['stim_nms']) if all([nm in ids_with_surprisal for nm in trial_nms])]
    # NOTE: might not need stimulus and response if we stop saving excessive copies of them in trf:
    keys_to_fitler=['stimulus','response','stim_nms']
    for k in keys_to_fitler:
        trf[k]=[trf[k][ii] for ii in indx_good_trials]
    # TODO: verify that we dont need to return anything...?
    # return trf_filtered    

def get_word_reconstruction_accuracies(trf,paired_surprisal_bounds):
    #TODO: call trf.predict to get predicted stim envelopes
    # then use to compute per-word correlation after segmenting 
    # real and predicted envelopes  
    
    def segment_trials_by_words(trf,paired_surprisal_bounds):
        '''
        TODO: this function might not be necessary...
        NOTE: I guess we don't need paired_surprisal_bounds here since still just looking at times
        1. take in trial_nms from trf (trf['stim_nms']), 
        2. lookup corresponding tg info from paired_surprisal_bounds
        3. add durations of preceding sentences 
        to tg info of later sentences in trial 
        4. package into trial_word_times
        '''
        fs=100 # NOTE: i'm pretty sure this is true but should confirm
        #TODO: account for mismatched word lists in paired_surprisal_bounds
        trial_nms=trf['stim_nms']        

        def get_trial_word_times(trial_nms,paired_surprisal_bounds):
            # TODO: this might not need to be it's own function really
            # unless we're planning to somehow vectorize the final correlation computation
            # in which case 
            n_trials=len(trial_nms)
            for nn_trial, names_in_trial in enumerate(trial_nms):
                # get words in trial
                trial_words=[paired_surprisal_bounds[nm]['boundaries']['words'] for nm in names_in_trial]
                #should be a list of lists where sublists: [onset,offset,word_str]
                n_words=sum([len(tw) for tw in trial_words])
                trail_word_times=np.nan(n_words,2)
                trial_words=[None]*n_words
                # get the durations of each sentence in trial, 
                # exclude the first one
                for ss, (stim_id,word_info) in enumerate(names_in_trial[1:],trial_words):
                    stim_dur=(trf['stimulus'][nn_trial][ss].size-1)/fs


            return (trial_word_times,trial_words)
        get_trial_word_times(trial_nms,paired_surprisal_bounds)
        return segmented_trials
    segment_trials_by_words(trf,paired_surprisal_bounds)
    pass

#%% TEST FUNCTION HERE:
#NOTE: we filtered out trials that are unusable bc no surprisal values but still need 
# to deal with cases whereno bounds (cuz words mismatched)
filter_trf_stimuli(trf,surprisal_ids)
get_word_reconstruction_accuracies(trf,paired_surprisal_bounds)
# checking what is contained in trfs that we saved because I forgot:
# for model in trf['trained_models']:
#     print(np.all(model.weights==trf['trf_fitted'].weights))
# seems like all the models have the same weights - this indicates they are not actually from 
# different data partitions in nested crossvalidation procedure
# ignoring this for now to continue with rest of the code
#TODO: choose optimal lambda value, train on entire dataset with that value, predict envelopes, 
# split (reconstructed AND true) envelopes along word onsets/offsets, pair with surprisal values, run lme model
#%% unfinished functions? 
#TODO: determine if necessary?
# based on the function names, I'm gathering that we might have figured out how to get the word timings 
# into a dictionary but not how to pair them with surprisal values
def get_sentence_from_surprisal_tup(surprisal_tup):
    '''
    given particular tuple from surprisals (sentence_list, surprisal_vals), extract just the sentence
    unlike boundaries, surprisals dont 
    '''
    pass
def get_bounds_for_sentence(boundaries,skey):
    
    return 
def match_short_nm(long_nm,short_nms):
    '''
    TODO: determine if unnecessary?
    helper function to match detailed/long name to corresponding short sor
    '''
    pass
def make_surprisal_vector(onsets,surprisal_vals,fs_output=100):
    '''
    onsets are in seconds relative to sentence beginning
    '''
    pass

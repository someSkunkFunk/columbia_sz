# script for mtrf analysis on columbia sz dataset using bluehive
# most up-to-date as of 9/26/23

#%%
# import packages
import pickle
import scipy.io as spio
import numpy as np
import os

import utils
from trf_helpers import find_bad_electrodes, get_stim_envs, setup_xy
from mtrf.model import TRF
from mtrf.stats import crossval, nested_crossval


#%%
# define wrapper,utilities
def sensible_lengths(seg_data:tuple,stim_data):
    '''
    helper fnc to filter out eeg data where durations are obviously wrong
    due to evnt timestamps
    stim data is assumed to be dictionary with stim envelopes, keys are stim nms
    seg_data is evnt-generated preprocessed data tuple with 
    (list of stim nms in segment, audio recodring array for segment, eeg segment array)
    returns True or False
    '''
    # NOTE: subj_data contains eeg data sampled at fs_trf
    # and audio recording at eeg_fs
    _fs=100
    _max_diff=0.5 # in seconds
    eeg_dur=(len(seg_data[-1])-1)/_fs #hard-coded fs here...
    stim_nms=[nm.strip(".wav") for nm in seg_data[0]]
    audio_dur=sum([stim_data[nm].size for nm in stim_nms])/_fs
    if abs(audio_dur-eeg_dur) < _max_diff:
        return True
    else:
        return False 


def nested_cv_wrapper(subj_num,
                      direction=-1,
                      tmin=0,
                      tmax=0.4,
                      k=5,
                      lim_stim=None,
                      save_results=False,
                      drop_bad_electrodes=False,
                      clean_nxor_noisy=['clean'], 
                      regs=np.logspace(-1, 8, 10),
                      reduce_trials_by="pauses",
                      return_xy=False, 
                      evnt=False,which_xcorr=None
                      ):
    '''
    NOTE: if using evnt timestamps, reduce_trials_by="pauses" will NOT work
    direction: 1 -> encoder -1 -> decoder
    lim_stim: limit to n number of stimuli for faster execution (won't save result)
    clean_nxor_noisy: non-exclusive or choose cleanr or noisy (list, turned into list if not entered as list but elements must be strings)
    drop_bad_electrodes: drop electrodes that are "outliers"
    regs: ridge regularization params to optimize over
    reduce_trials_by: str specifying  by grouping stories wihtin a block ("stim_nm")
    # or grouping by pauses within a block ("pauses")
    '''
    f_lp=49 #Hz, lowpass filter freq for get_stim_envs
    subj_cat=utils.get_subj_cat(subj_num)
    #NOTE that  by leaving eeg_dir blank below it's looking 
    # in eeg_data/preproessed_xcorr by default
    subj_data=utils.load_preprocessed(subj_num,evnt=evnt,which_xcorr=which_xcorr)
    if evnt:
        print(f"Evnt preprocessed data loaded.")
        # evnt data has ([stim_nms],np.arr[normalized_aud],np.arr[eeg])
        fs_trf=100 #TODO: something about this
        #NOTE: reduction by pauses may still be applicable here but for now
        # just going to worry about removing evnt stims with nonsensical durations
        
        reduce_trials_by=None # already "reduced" by pauses
    else:
        print(f"{which_xcorr} xcorr preprocessed data loaded.")
        fs_trf=subj_data['fs'][0]
    # specify fl paths assumes running from code as pwd
    eeg_dir=os.path.join("..","eeg_data")
    # stim_fnm = "master_stim_file_schiz_studybis.mat" # note this is original fnm from box, we changed to just stim_info.mat
    stim_fl_path=os.path.join(eeg_dir,"stim_info.mat")
    stims_dict=utils.get_stims_dict(stim_fl_path)
    
    if lim_stim is not None:
        # in case we want to run to completion for testing 
        print(f'running number of stimuli limited to {lim_stim}, wont save result...\n')
    if (lim_stim is not None) and save_results:
        raise NotImplementedError("not saving results by default because assuming we just want to test that this runs.") 

    if isinstance(clean_nxor_noisy, str):
        # if single string given instead of list
        clean_nxor_noisy = [clean_nxor_noisy]
          # 
    if drop_bad_electrodes:
        outlier_idx=find_bad_electrodes(subj_data)
    else:
        outlier_idx=None

    for clean_or_noisy in clean_nxor_noisy:
        #TODO: pre-compute the stim envelopes before running trf analysis
        # so they can just be loaded rather than waiting for computing each
        print(f"clean_or_noisy: {clean_or_noisy};\nnot used here anymore since pre-computed, but this is what was used to generate stim_envs")
        # stim_envs=get_stim_envs(stims_dict,clean_or_noisy,fs_output=fs_trf,f_lp=f_lp)
        # save_pth=os.path.join("..","eeg_data","stim_envs.pkl")
        # with open(save_pth,'wb') as fl:
        #     pickle.dump(stim_envs,fl)
        #     print(f"saved stim_envs to {save_pth}")
        stim_envs_pth=os.path.join("..","eeg_data","stim_envs.pkl")
        with open(stim_envs_pth, 'rb') as fl:
            stim_envs=pickle.load(fl)
            

        #recorded audio mostly for debuggning and checking alignment of timestamps
        #TODO: fix setupxy so it works with evnt stamps
        if evnt:
            # clean subj_data trials where stim/response durations are too different
            subj_data={seg_nm:seg_data for seg_nm,seg_data in subj_data.items() if sensible_lengths(seg_data,stim_envs)}
                

        stimulus,response,stim_nms,recorded_audio=setup_xy(subj_data,stim_envs,
                                                subj_num,reduce_trials_by,
                                                outlier_idx,evnt=evnt,which_xcorr=which_xcorr)
        # model params      
        trf = TRF(direction=direction)  # use bkwd model
        


        r_ncv, best_lam=nested_crossval(trf, stimulus[:lim_stim], response[:lim_stim], fs_trf, tmin, tmax, regs, k=k)
        print(f"r-values: {r_ncv}, mean: {r_ncv.mean()}")


        if lim_stim is None and save_results:
            # save results

            results_file = "bkwd_trf.pkl"
            if evnt:
                timestamps_generated_by="evnt"
                # results_dir = os.path.join("..","evnt_results", subj_cat, subj_num)
            else:
                timestamps_generated_by=f"xcorr{which_xcorr}"
                # results_dir = os.path.join("..",f"{which_xcorr}_results", subj_cat, subj_num)
            results_dir=os.path.join("..","results",subj_cat,subj_num)
            
            if reduce_trials_by is not None:
                trial_reduction=reduce_trials_by
            else:
                trial_reduction="None"
            # Check if the directory exists; if not, create it
            # note: will also create parent directoriesr
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            results_pth=os.path.join(results_dir, results_file)
            print(f"saving results to {results_pth}")
            with open(results_pth, 'wb') as f:
                pickle.dump({'trf_fitted': trf, 'r_ncv': r_ncv, 'best_lam': best_lam,
                                'stimulus': stimulus, 'response': response, 'stim_nms': stim_nms,
                                'trials_reduced_by':trial_reduction,
                                'timestamps_generated_by':timestamps_generated_by}, f)


    if return_xy == True:
        return trf, r_ncv, best_lam, (stimulus, response, stim_nms)
    else:
        return trf, r_ncv, best_lam
#%%
if __name__=="__main__":
    
    if "which_stmps" in os.environ and "subj_num" in os.environ:
        subj_num=os.environ["subj_num"] 
        which_stmps=os.environ["which_stmps"]
        
        if which_stmps.lower()=="xcorr":
            evnt=False
            which_xcorr=os.environ["which_xcorr"]
        elif which_stmps.lower()=="evnt":
            evnt=True #IF TRUE USE EVNT-SEGMENTED DATA
            which_xcorr=None #TODO: this is already default in load_preprocessed but I'm specifying it like 6 different places, how can I avoid this?
        # code is getting really messy because of this stupidass variable, it is now in setup_xy, env_recon, get_pause_times, and anywhere timestamps are used
        else:
            raise NotImplementedError(f"which_stmps={which_stmps} is not an option")
    else:
        # running interactively probably for debugging purposes
        #NOTE: why do I need which_stamps AND evnt??
        #RE: seems like which_stmps used in bash script, automate in interactive more
        subj_num="3253"
        evnt=True
        # subj_cat=utils.get_subj_cat(subj_num)
        if evnt:
            which_xcorr=None
        else:
            which_xcorr="wavs"
        
    #note: return_xy is False by default but when save_results is True will store them in pkl anyway
    nested_cv_wrapper(subj_num,return_xy=False,
                      save_results=True,
                      evnt=evnt,which_xcorr=which_xcorr)

                     


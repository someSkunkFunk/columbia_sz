# script for bkwd mtrf analysis 
#%%
# INIT
import pickle
import scipy.io as spio

import numpy as np
import os

import utils
from trf_helpers import find_bad_electrodes, load_stim_envs, setup_xy
from mtrf.model import TRF
from mtrf.stats import crossval, nested_crossval

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

def filter_blocks_idx(blocks_to_keep,stim_nms):
    btk_set=set(blocks_to_keep)
    blocks_idx=[idx for idx,nms in enumerate(stim_nms) if nms[0][:3] in btk_set]
    return blocks_idx 
def filter_lists_with_indices(lists_to_filter, filtered_indices):
    filtered_lists = [[inner[idx] for idx in filtered_indices] for inner in lists_to_filter]
    return filtered_lists

def make_debug_plots(eeg_dir,stim_nms,stimulus,clean_stimulus,
                     recorded_audio,fs_trf,subj_num,subj_cat,thresh_dir):
    import matplotlib.pyplot as plt
    
    stim_fl_path=os.path.join(eeg_dir, "stim_info.mat")
    stims_dict=utils.get_stims_dict(stim_fl_path)
    
    fs_audio=stims_dict['fs'][0]
    fs_rec=2400

    for stim_ii, (stim_input,clean_stim_input,recording,nms) in enumerate(zip(stimulus,clean_stimulus,recorded_audio,stim_nms)):
        print(f"plotting stim {stim_ii} of {len(stimulus)}")
        noisy_stim_wav=np.concatenate([utils.get_stim_wav(stims_dict,stim_nm,'noisy') for stim_nm in nms])
        clean_stim_wav=np.concatenate([utils.get_stim_wav(stims_dict,stim_nm,'clean') for stim_nm in nms])
        # stim_input=stimulus[stim_ii]

        # recording=recorded_audio[stim_ii]
        t_rec=np.arange(recording.size)/fs_rec
        t_stim=np.arange(noisy_stim_wav.size)/fs_audio
        t_trf_input=np.arange(stim_input.size)/fs_trf
        fig,ax=plt.subplots(4,1,figsize=[12,10])
        ax[0].plot(t_rec,recording,label="eeg audio")
        ax[0].plot(t_stim,noisy_stim_wav,label="noisy stim")
        ax[0].legend()
        ax[0].set_title(f'noisy {nms} wavs and eeg recorded audio')
        # plt.show()

        ax[1].plot(t_rec,recording,label="eeg audio")
        ax[1].plot(t_stim,clean_stim_wav,label="clean stim")
        ax[1].legend()
        ax[1].set_title(f'clean {nms} wav and eeg recorded audio')
        # plt.show()

        ax[2].plot(t_stim,noisy_stim_wav,label="noisy stim")
        ax[2].plot(t_trf_input,stim_input,label="noisy trf input")
        ax[2].legend()
        ax[2].set_title(f'noisy {nms} wav and noisy trf input envelope')
        # plt.show()
        ax[3].plot(t_stim,clean_stim_wav,label="clean stim")
        ax[3].plot(t_trf_input,clean_stim_input,label="clean trf input")
        ax[3].legend()
        ax[3].set_title(f'clean {nms} wav and clean trf input envelope')
        # plt.show()

        # raise NotImplementedError('temporary pause here')
        stim_num=f'{stim_ii:03}'

        fig_dir=os.path.join("..","figures","trf_input_align_check_decimate",
                             thresh_dir,subj_cat,subj_num)
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir,exist_ok=True)
        fig_pth=os.path.join(fig_dir,stim_num)
        plt.savefig(fig_pth)
        plt.close()
        del fig, ax
def nested_cv_wrapper(subj_num,
                      direction=-1,
                      tmin=-0.1,
                      tmax=0.4,
                      k=-1,
                      lim_stim=None,
                      save_results=False,
                      drop_bad_electrodes=False,
                      clean_nxor_noisy=['noisy'], 
                      regs=np.logspace(-1, 8, 10),
                      reduce_trials_by="pauses",
                      return_xy=False, 
                      evnt=False,which_xcorr=None,
                      evnt_thresh=None,
                      shuffle_trials=True,
                      blocks='all',
                      cv_method='crossval',
                      which_envs=None
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
    print(f"running blocks: {blocks}")
    # f_lp=49 #Hz, lowpass filter freq for get_stim_envs
    subj_cat=utils.get_subj_cat(subj_num)
    # specify fl paths assumes running from code as pwd
    eeg_dir=os.path.join("..","eeg_data")
    if evnt:
        thresh_dir=f"thresh_{evnt_thresh}" # should be three digit number representing decimals to third place
        prep_data_dir=os.path.join(eeg_dir,'preprocessed_decimate',thresh_dir)
        subj_data=utils.load_preprocessed(subj_num,eeg_dir=prep_data_dir,
                                          evnt=evnt,which_xcorr=which_xcorr)
    else:
        # subj_data=utils.load_preprocessed(subj_num,evnt=evnt,which_xcorr=which_xcorr) 
        raise NotImplementedError("Loading preprocessed xcorr generated data needs different directory, although I think the default commented above will work?")
    # stim_fnm = "master_stim_file_schiz_studybis.mat" # note this is original fnm from box, we changed to just stim_info.mat
    # stim_fl_path=os.path.join(eeg_dir,"stim_info.mat")
    # stims_dict=utils.get_stims_dict(stim_fl_path)
    #NOTE that  by leaving eeg_dir blank below it's looking 
    # in eeg_data/preproessed_xcorr by default
    if shuffle_trials:
        shuffled="shuffled"
    else:
        shuffled="shuffless"
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
    
    
    if lim_stim is not None:
        # in case we want to run to completion for testing 
        print(f'running number of stimuli limited to {lim_stim}, wont save result...\n')
        print(' NOTE THIS IS ACTUALLY NOT DOING ANYTHING')
    if (lim_stim is not None) and save_results:
        print("not saving results by default because assuming we just want to test that this runs.") 

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
        _stim_lowpass_f='49'
        # stim_envs=get_stim_envs(stims_dict,clean_or_noisy,fs_output=fs_trf,f_lp=f_lp)
        # save_pth=os.path.join("..","eeg_data","stim_envs.pkl")
        # with open(save_pth,'wb') as fl:
        #     pickle.dump(stim_envs,fl)
        #     print(f"saved stim_envs to {save_pth}")
        if which_envs=='rms':
            which_envs_str="_rms_"
            stim_envs=utils.load_matlab_envs(clean_or_noisy)
        else:
            print(f"loading pre-computed envelopes for {clean_or_noisy} stims lowpassed at {_stim_lowpass_f};")
            which_envs_str=""
            stim_envs=load_stim_envs(lowpass_f=_stim_lowpass_f,clean_or_noisy=clean_or_noisy)
        

        #recorded audio mostly for debuggning and checking alignment of timestamps
        #TODO: fix setupxy so it works with evnt stamps
        if evnt:
            # clean subj_data trials where stim/response durations are too different
            num_trials_pre=len(subj_data)
            subj_data={seg_nm:seg_data for seg_nm,seg_data in subj_data.items() if sensible_lengths(seg_data,stim_envs)}
            num_trials_post=len(subj_data)
            print(f"removed {(num_trials_pre-num_trials_post)} trials due to inconsistent stim,resp lengths.")
                

        stimulus,response,stim_nms,recorded_audio=setup_xy(subj_data,stim_envs,
                                                subj_num,reduce_trials_by,
                                                outlier_idx,evnt=evnt,which_xcorr=which_xcorr,
                                                shuffle_trials=shuffle_trials)
        # should disable plotting when running actual slurm job
        if "which_stmps" in os.environ:
            # NOTE: set to False to avoid short-cutting SLURM jobs where we want envelope recons to happen
            _debug_alignment=False
        else:
            print("DEBUG ALIGNMENT AUTOMATICALLY SET TO TRUE HERE")
            _debug_alignment=True
        if _debug_alignment:
            print(f"DEBUG ALIGN ENABLED, NOT DOING TRF, JUST PLOTTING STIMULI INPUTS AND RECORDED AUDIO")
            clean_envs=utils.load_matlab_envs('clean')
            #unumute to load the ones computed using python:
            # clean_envs=load_stim_envs(lowpass_f=_stim_lowpass_f,clean_or_noisy='clean',norm=True)
            clean_stimulus,_,_,_=setup_xy(subj_data,clean_envs,
                                                subj_num,reduce_trials_by,
                                                outlier_idx,evnt=evnt,which_xcorr=which_xcorr,
                                                shuffle_trials=shuffle_trials)
            make_debug_plots(eeg_dir,stim_nms,stimulus,clean_stimulus,recorded_audio,fs_trf,subj_num,subj_cat,thresh_dir)
            print("done plotting")
        else:
            if blocks!='all' and blocks!='1,2,3,4,5,6':
                #filter blocks 
                # add strings to match stim_nms
                blocks_to_keep=['b0'+b.strip() for b in blocks.split(",")]
                blocks_to_keep_idx=filter_blocks_idx(blocks_to_keep,stim_nms)
                [stimulus,response,stim_nms]=filter_lists_with_indices([stimulus,response,stim_nms],blocks_to_keep_idx)
                
            total_sound_time=sum([len(s)/fs_trf for s in stimulus])
            total_response_time=sum([len(r)/fs_trf for r in response])
            print(f"total stim time: {total_sound_time}\ntotal response time: {total_response_time}")
            # init bkwd model
            trf = TRF(direction=direction)  

            if cv_method.lower()=='nested':
                print(f"using k={k} folds for nested cross validations")
                r_ncv, best_lam=nested_crossval(trf, stimulus[:lim_stim], response[:lim_stim], fs_trf, tmin, tmax, regs, k=k)
                print(f"r-values: {r_ncv}, mean: {r_ncv.mean()}, best_lam:{best_lam}")
            elif cv_method.lower()=='crossval':
                regularization=0
                print(f"using k={k} folds for nested cross validations, with regularization={regularization}")
                r_mean=crossval(trf, stimulus[:lim_stim], response[:lim_stim], fs_trf, tmin, tmax, regularization, k=k)
                print(f"mean r-val: {r_mean}")



            if lim_stim is None and save_results:
                
                # save results
            
                results_file= f"bkwd_trf_{clean_or_noisy}{which_envs_str}stims_{cv_method}.pkl"
                # note the clean ones didn't specify in file name since added string formatting after
                # but whatever
                if evnt:
                    timestamps_generated_by="evnt"
                    if k!=-1:
                        thresh_folds_dir=thresh_dir+f"_{k}fold"+f"_{shuffled}"
                    elif k==-1:
                        thresh_folds_dir=thresh_dir+"_loo"+f"_{shuffled}"
                    if blocks!="all" and blocks!="1,2,3,4,5,6":
                        print("note: need to change this in xcorr case")
                        blocks_str="".join(blocks_to_keep)
                        thresh_folds_dir=thresh_folds_dir+"_"+blocks_str
                    results_dir=os.path.join("..","results","evnt_decimate",
                                            thresh_folds_dir,subj_cat,subj_num)
                    # results_dir = os.path.join("..","evnt_results", subj_cat, subj_num)
                else:
                    timestamps_generated_by=f"xcorr{which_xcorr}"
                    xcorr_subdir=f"xcorr_{which_xcorr}"
                    results_dir = os.path.join("..","results",xcorr_subdir,subj_cat,subj_num)
                
                
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
                    if cv_method.lower()=='nested':
                        pickle.dump({'trf_fitted': trf, 'r_ncv': r_ncv, 'best_lam': best_lam,
                                        'stimulus': stimulus, 'response': response, 'stim_nms': stim_nms,
                                        'trials_reduced_by':trial_reduction,
                                        'timestamps_generated_by':timestamps_generated_by}, f)
                    elif cv_method.lower()=='crossval':
                        pickle.dump({'trf_fitted': trf, 'r_mean': r_mean, 'regularization': regularization,
                                        'stimulus': stimulus, 'response': response, 'stim_nms': stim_nms,
                                        'trials_reduced_by':trial_reduction,
                                        'timestamps_generated_by':timestamps_generated_by}, f)
                

    #NOTE: disabling return functionality since we never use it and causes errors now when cv_method is not nested
    if return_xy == True:
        pass
        # return trf, r_ncv, best_lam, (stimulus, response, stim_nms)
    else:
        pass
        # return trf, r_ncv, best_lam
#%% MAIN SCRIPT
if __name__=="__main__":
    
    if "which_stmps" in os.environ:
        subj_num=utils.assign_subj(os.environ["SLURM_ARRAY_TASK_ID"]) 
        which_stmps=os.environ["which_stmps"]
        k=int(os.environ["k_folds"])
        bool_dict={'true':True,'false':False}
        shuffle_trials=bool_dict[os.environ["shuffle_trials"].lower()]
        blocks=os.environ["blocks"]
        cv_method=os.environ["cv_method"]
        which_envs=os.environ["which_envs"]

        
        if which_stmps.lower()=="xcorr":
            evnt=False
            which_xcorr=os.environ["which_xcorr"]
        elif which_stmps.lower()=="evnt":
            evnt_thresh=os.environ["evnt_thresh"]
            print(f"evnt_thresh selected: {evnt_thresh}")
            evnt=True #IF TRUE USE EVNT-SEGMENTED DATA
            which_xcorr=None #TODO: this is already default in load_preprocessed but I'm specifying it like 6 different places, how can I avoid this?
        # code is getting really messy because of this stupidass variable, it is now in setup_xy, env_recon, get_pause_times, and anywhere timestamps are used
        else:
            raise NotImplementedError(f"which_stmps={which_stmps} is not an option")
    else:
        # running interactively probably for debugging purposes
        #NOTE: why do I need which_stamps AND evnt??
        #RE: seems like which_stmps used in bash script, automate in interactive more
        subj_num="3317"
        evnt=True
        evnt_thresh="750"
        k=5
        shuffle_trials=False
        blocks='all'
        cv_method="nested"
        # lim_stim=50
        print(f"evnt_thresh selected: {evnt_thresh}")

        # subj_cat=utils.get_subj_cat(subj_num)
        if evnt:
            which_xcorr=None
        else:
            which_xcorr="wavs"
        which_envs='rms'
        
        
    #note: return_xy is False by default but when save_results is True will store them in pkl anyway
    print(f"running subject {subj_num}...")
    nested_cv_wrapper(subj_num,save_results=True,
                    evnt=evnt,
                    evnt_thresh=evnt_thresh,
                    which_xcorr=which_xcorr,
                    k=k,shuffle_trials=shuffle_trials,
                    blocks=blocks,cv_method=cv_method,
                    which_envs=which_envs)
    print(f"{subj_num} TRF complete.")

                     



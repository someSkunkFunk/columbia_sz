# script for getting optimization tuning curve for regularization parameter lambda
#%%
# INIT
import pickle
import numpy as np
import os

import utils
from trf_helpers import find_bad_electrodes, load_stim_envs, setup_xy, sensible_lengths
from mtrf.model import TRF
from mtrf.stats import crossval
def get_tuning_curve(subj_num,
                      direction=-1,
                      tmin=-0.1,
                      tmax=0.4,
                      k=-1,
                      lim_stim=None,
                      save_results=False,
                      drop_bad_electrodes=False,
                      clean_nxor_noisy=['clean','noisy'], 
                      regs=np.logspace(-10, 12, 50),
                      reduce_trials_by="pauses",
                      return_xy=False,
                      evnt=False,which_xcorr=None,
                      evnt_thresh=None,
                      shuffle_trials=True,
                      blocks='all',
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
    print(f"stimuli envelopes to use as input: {clean_nxor_noisy}")
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
    # generate mathching thresh-folds dir 
    # (note: maybe make this a function called in both og_env_recon and here so names are consistent)
    if k!=-1:
        thresh_folds_dir=thresh_dir+f"_{k}fold"+f"_{shuffled}"
    elif k==-1:
        thresh_folds_dir=thresh_dir+"_loo"+f"_{shuffled}"
    if blocks!="all" and blocks!="1,2,3,4,5,6":
        print("note: need to change this in xcorr case")
        blocks_str="".join(blocks_to_keep)
        thresh_folds_dir=thresh_folds_dir+"_"+blocks_str
    for clean_or_noisy in clean_nxor_noisy:
        print(f'running {clean_or_noisy}...')
        if which_envs=='rms':
            print("Loading MATLAB computed envelopes")
            which_envs_str="_rms"
            stim_envs=utils.load_matlab_envs(clean_or_noisy)
        else:
            _stim_lowpass_f='49'
            print(f"loading pre-computed envelopes for {clean_or_noisy} stims lowpassed at {_stim_lowpass_f};")
            which_envs_str=""
            stim_envs=load_stim_envs(lowpass_f=_stim_lowpass_f,clean_or_noisy=clean_or_noisy)
        

        #recorded audio mostly for debuggning and checking alignment of timestamps
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
            
            _debug_alignment=False
            if _debug_alignment:
                print('Set debug alignent to true in SLURM job... will plott all stimuli and recorded audio')
        else:
            # print("SET DEBUG ALIGNMENT TO TRUE HERE FOR PLOTTING DURING INTERACTIVE SESSION")
            _debug_alignment=False
        if _debug_alignment:
            print(f"DEBUG ALIGN ENABLED, NOT DOING TRF")
            # clean_envs=load_stim_envs(lowpass_f=_stim_lowpass_f,clean_or_noisy='clean',norm=True)
            if clean_or_noisy=='noisy':
                other_envs=utils.load_matlab_envs('clean')
            else:
                other_envs=utils.load_matlab_envs('noisy')

        
        if blocks!='all' and blocks!='1,2,3,4,5,6':
            #filter blocks 
            # add strings to match stim_nms
            blocks_to_keep=['b0'+b.strip() for b in blocks.split(",")]
            blocks_to_keep_idx=utils.filter_blocks_idx(blocks_to_keep,stim_nms)
            [stimulus,response,stim_nms]=utils.filter_lists_with_indices([stimulus,response,stim_nms],blocks_to_keep_idx)
            
        total_sound_time=sum([len(s)/fs_trf for s in stimulus])
        total_response_time=sum([len(r)/fs_trf for r in response])
        print(f"total stim time: {total_sound_time}\ntotal response time: {total_response_time}")
        results=[('regularization', 'mean_r','trf_model')]#store (regularization, r-val, trf_model)
        for regularization in regs:
            # init bkwd model
            trf=TRF(direction=direction)  

        
            print(f"using k={k} folds for cross validation, with regularization={regularization}")
            r_mean=crossval(trf, stimulus[:lim_stim], response[:lim_stim], fs_trf, tmin, tmax, regularization, k=k)
            print(f"mean r-val: {r_mean}")
            results.append((regularization,r_mean,trf))
        
        
        results_dir=os.path.join("..","results","evnt_decimate",thresh_folds_dir,subj_cat,subj_num)
        results_fnm=f"lambda_tuning_curve_{clean_or_noisy}{which_envs_str}.pkl"
        results_path=os.path.join(results_dir,results_fnm)
        print(f"Storing (regularization, r-val, trf_model) to {results_path}")
        with open(results_path, 'wb') as file:
            pickle.dump(results, file)
        print(f"{subj_num} complete.")
            
#%% MAIN SCRIPT
if __name__=="__main__":
    
    if "which_stmps" in os.environ:
        subj_num=utils.assign_subj(os.environ["SLURM_ARRAY_TASK_ID"]) 
        which_stmps=os.environ["which_stmps"]
        k=int(os.environ["k_folds"])
        bool_dict={'true':True,'false':False}
        shuffle_trials=bool_dict[os.environ["shuffle_trials"].lower()]
        blocks=os.environ["blocks"]
        which_envs=os.environ["which_envs"]
        # clean_or_noisy=os.environ["clean_or_noisy"]


        
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
        subj_num="3316"
        evnt=True
        evnt_thresh="750"
        k=5
        shuffle_trials=True
        blocks='all'
        clean_or_noisy="noisy"
        # lim_stim=50
        print(f"evnt_thresh selected: {evnt_thresh}")

        # subj_cat=utils.get_subj_cat(subj_num)
        if evnt:
            which_xcorr=None
        else:
            which_xcorr="wavs"
        which_envs='rms'
    print(f"running subject {subj_num}...")
    get_tuning_curve(subj_num,save_results=True,
                    evnt=evnt,
                    evnt_thresh=evnt_thresh,
                    which_xcorr=which_xcorr,
                    k=k,shuffle_trials=shuffle_trials,
                    blocks=blocks,
                    which_envs=which_envs)
    print(f"{subj_num} tuning curve complete.")

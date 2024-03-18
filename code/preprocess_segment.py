# Already have stim timestamps for each stim in some subjects; for those
# we just preprocess and segment
# for subjects without timestamps, first get the timestamps, then preprocess and segment
#%%
# imports, etc.
import pickle
import scipy.io as spio
import numpy as np

import os
# import sounddevice as sd # note not available via conda....? not sure I'll need anyway so ignore for now
from scipy import signal
import matplotlib.pyplot as plt
import utils

# specify fl paths assumes running from code as pwd
eeg_dir=os.path.join("..", "eeg_data")
# stim_fnm = "master_stim_file_schiz_studybis.mat" # note this is original fnm from box, we changed to just stim_info.mat
stim_fnm="stim_info.mat"
stim_fl_path=os.path.join(eeg_dir, "stim_info.mat")
stims_dict=utils.get_stims_dict(stim_fl_path)
fs_audio=stims_dict['fs'][0] # 11025 foriginally
# fs_audio=16000 #just trying it out cuz nothing else works
fs_eeg=2400 #trie d2kHz didn't help
fs_trf=100 # Hz, downsampling frequency for trf analysis
n_blocks = 6
# blocks=["B2"]
blocks = [f"B{ii}" for ii in range(1, n_blocks+1)]
print(f"check blocks: {blocks}")
#%%
# setup
# bash script vars
# 
###########################################################################################
if "subj_num" in os.environ: 
    subj_num=os.environ["subj_num"]
    which_stmps=os.environ["which_stmps"] #xcorr or evnt
    which_xcorr=os.environ["which_xcorr"]
    #bool vars need to be converted from strings
    bool_dict={"true":True,"false":False}
    do_avg_ref=bool_dict[os.environ["do_avg_ref"].lower()]
    print(f"do_avg_ref: {do_avg_ref}")
    just_stmp=bool_dict[os.environ["just_stmp"].lower()]
    print(f"just_stamp translated into: {just_stmp}")
#####################################################################################
#manual vars
#####################################################################################
else:
    print("using manually inputted vars")
    subj_num="3218"
    which_stmps="evnt"
    which_xcorr="envs"
    just_stmp=False
    do_avg_ref=False
    noisy_or_clean="noisy" #NOTE: clean is default and setting them here does nothing
##################################################################################
thresh_params=('pearsonr', 0.4)
# cutoff_ratio=10

#print  correlation thresholds
if thresh_params[0]=='xcorr_peak':
    print(f'using xcorr peak thresholding; xcorr cutoff: {thresh_params[1]} * std(xcorr)')
elif thresh_params[0]=='pearsonr':
    print(f"using pearsonr thresholding; pearsonr threshold: {thresh_params[1]}")
timestamps_bad=True #currently unsure where the problem is but could still be timestamps although they seem good
# determine filter params applied to EEG before segmentation 
# NOTE: different from filter lims used in timestamp detection algo (!)
filt_band_lims=[1.0, 15] #Hz; highpass, lowpass cutoffs
filt_o=3 # order of filter (effective order x2 of this since using zero-phase)
processed_dir_path=os.path.join(eeg_dir, f"preprocessed_{which_stmps}") #directory where processed data goes
subj_cat=utils.get_subj_cat(subj_num) #note: checked get_subj_cat, should be fine
raw_dir=os.path.join(eeg_dir,"raw")
print(f"Fetching data for {subj_num,subj_cat}")
subj_eeg=utils.get_full_raw_eeg(raw_dir,subj_cat,subj_num,blocks=blocks)
#%%
# find timestamps
if which_stmps=="xcorr":
    # Find timestamps using xcorr algo
    # check if save directory exists, else make one
    save_path=os.path.join(processed_dir_path,subj_cat,subj_num)
    if not os.path.isdir(save_path):
        print(f"preprocessed dir for {subj_num, subj_cat} not found, creating one.")
        os.makedirs(save_path, exist_ok=True)
    # check if timestamps fl exists already
    timestamps_path=os.path.join("..","eeg_data","timestamps",subj_cat,subj_num,
                                f"{which_xcorr}_timestamps.pkl")
    if os.path.exists(timestamps_path) and not timestamps_bad:
        # if already have timestamps, load from pkl:
        print(f"{subj_num, subj_cat} already has timestamps, loading from pkl.")
        with open(timestamps_path, 'rb') as pkl_fl: 
            timestamps = pickle.load(pkl_fl)
    else:
        print(f"Generating timestamps for {subj_num, subj_cat} ...")
        # get timestamps
        timestamps=utils.get_timestamps(subj_eeg,raw_dir,subj_num,
                                        subj_cat,stims_dict,blocks,
                                        thresh_params,which_xcorr)
        # check resulting times
        if timestamps=={}:
            print("timestamps dictionary is empty.")
        else:
            total_soundtime=0
            missing_stims_list=[]
            for block in timestamps:
                block_sound_time=0
                for stim_nm, (start, end) in timestamps[block].items():
                    if all([start,end]):
                        block_sound_time+=(end-start-1)/fs_eeg
                    else:
                        missing_stims_list.append(stim_nm)
                print(f"in block {block}, total sound time is {block_sound_time:.3f} s.")
                total_soundtime+=block_sound_time
            print(f"total sound time: {total_soundtime:.3f} s.")
            print(f"missing stims:\n{len(missing_stims_list)}")
            #  save stim timestamps
            with open(timestamps_path, 'wb') as f:
                print(f"saving timestamps for {subj_num}")
                pickle.dump(timestamps, f)
if which_stmps=="evnt":
    # load evnt timestamps
    #TODO: make evnt timestamps loading function a util
    # check if save directory exists, else make one
    save_path = os.path.join(processed_dir_path, subj_cat, subj_num)
    if not os.path.isdir(save_path):
        print(f"preprocessed dir for {subj_num, subj_cat} not found, creating one.")
        os.makedirs(save_path, exist_ok=True)
    # find evnt timestamps
    timestamps_path=os.path.join("..","eeg_data","timestamps",f"evnt_{subj_num}.mat")
    evnt_mat=spio.loadmat(timestamps_path)
    # returns dict for some reason, which mat2dict doesnt like
    evnt=evnt_mat['evnt']
    evnt=utils.mat2dict(evnt)
#%%
# preprocess each block separately
    
if not just_stmp:
    if which_stmps.lower()=='xcorr':
        timestamps_ds = {}
        print(f"Starting preprocessing {subj_num, subj_cat}")
        for block, raw_data in subj_eeg.items():
            #TODO: modify preprocessing block so that audio recording channel not filtered and can be used to check for alignment issues
            print(f"block: {block}")
            timestamps_ds[block] = {}
            # filter and resample
            raw_eeg=raw_data[:,:62]
            if do_avg_ref:
                print("Re-referencing eeg to common average.")
                raw_eeg=raw_eeg-raw_eeg.mean(axis=1)[:,None]
            if fs_eeg / 2 <= fs_trf:
                raise NotImplementedError("Nyquist") 
            sos=signal.butter(filt_o,filt_band_lims,btype='bandpass',
                            output='sos',fs=fs_eeg)
            filt_eeg=signal.sosfiltfilt(sos,raw_eeg,axis=0)
            audio_rec=raw_data[:,-1] # leave unperturbed for alignment checking
            # get number of samples in downsampled waveform
            num_ds=int(np.floor((filt_eeg.shape[0]-1)*(fs_trf/fs_eeg)))
            # downsample eeg
            # NOTE: audio_rec not downsampled
            subj_eeg[block]=(signal.resample(filt_eeg,num_ds,axis=0),audio_rec)
            # downsample timestamps
            for stim_nm, (start, end) in timestamps[block].items():
                if all([start,end]):
                    
                    s_ds=int(np.floor(start*(fs_trf/fs_eeg)))
                    e_ds=int(np.floor(end*(fs_trf/fs_eeg))) #NOTE: off by one error maybe?
                    timestamps_ds[block][stim_nm]=(s_ds, e_ds)
                else:
                    timestamps_ds[block][stim_nm]=(None, None)
        #%
        # align downsampled eeg using ds timestamps
        print(f"Preprocessing done for {subj_num, subj_cat}. algining and segmenting eeg")
        subj_data = utils.align_responses(subj_eeg, (timestamps, timestamps_ds), 
                                        audio_rec, stims_dict)
        subj_data['fs'] = fs_trf
        print("subj_data before pickling:")
        print(subj_data.head())
        print(f'saving to: {save_path}')
        subj_data.to_pickle(os.path.join(save_path, f"{which_xcorr}_aligned_resp.pkl"))
        print(f"{subj_num, subj_cat} preprocessing and segmentation complete.")
        # break
    elif which_stmps.lower()=='evnt':
        print(f"using evnt timestamps to segment eeg.")
        print("extracting evnt data...")
        evnt_nms=np.array([nm for arr_nm in evnt['name'][0] for nm in arr_nm])
        evnt_blocks=[nm[:3].capitalize() for nm in evnt_nms]
        evnt_confidence=np.array([float(s[0,0]) for s in evnt['confidence'][0]])
        evnt_onsets=fs_eeg*np.array([float(s[0,0]) for s in evnt['startTime'][0]])
        evnt_offsets=fs_eeg*np.array([float(s[0,0]) for s in evnt['stopTime'][0]])
        are_integer_valued=np.allclose(np.concatenate([evnt_onsets,evnt_offsets]), np.round(np.concatenate([evnt_onsets,evnt_offsets])))
        if are_integer_valued:
            evnt_onsets=evnt_onsets.astype(int)
            evnt_offsets=evnt_offsets.astype(int)
            print(f"onsets and offsets converted to integers: {are_integer_valued}")
        else:
            print(f"some values in start_times,end_times are not integer valued!")
        
        # split into blocks 
          
        for block, raw_data in subj_eeg.items():
            print(f"segmenting block {block}...")
            block_idx=[eb.replace('0','')==blocks[0] for eb in evnt_blocks]
            block_idx=np.array(block_idx,dtype=bool)
            stim_nms=evnt_nms[block_idx]
            confidence=evnt_confidence[block_idx]
            onsets=evnt_onsets[block_idx]
            offsets=evnt_offsets[block_idx]
            # filter and resample
            raw_eeg=raw_data[:,:62]
            audio_rec=raw_data[:,-1] # leave unperturbed for alignment checking
            
            if do_avg_ref:
                print("Re-referencing eeg to common average.")
                raw_eeg=raw_eeg-raw_eeg.mean(axis=1)[:,None]
            if fs_eeg / 2 <= fs_trf:
                raise NotImplementedError("Nyquist") 
            sos=signal.butter(filt_o,filt_band_lims,btype='bandpass',
                            output='sos',fs=fs_eeg)
            filt_eeg=signal.sosfiltfilt(sos,raw_eeg,axis=0)
            # BELOW THIS LINE, CHANGE EVNT REFERENCES TO BLOCK REFERENCES
            # prune low-confidence values,record figure for posterity:
            conf_thresh=0.0
            import matplotlib.pyplot as plt
            plt.hist(confidence,bins=50)
            plt.title(f"{subj_num}, {block} Evnt confidence vals")
            save_pth=os.path.join("..","figures","evnt_info",f"{subj_num}_{block}_confidence_hist.png")
            if not os.path.isdir(os.path.dirname(save_pth)):
                os.makedirs(os.path.dirname(save_pth),exist_ok=True)
            plt.axvline(conf_thresh,label=f'confidence threshold: {conf_thresh}')
            plt.savefig(save_pth)
            plt.show()
            plt.close()
            high_confidence=np.flatnonzero(confidence>conf_thresh)
            print(f"trimming {confidence.size-high_confidence.size} low confidence stims.")
            onsets=onsets[high_confidence]
            offsets=offsets[high_confidence]
            confidence=confidence[high_confidence]
            #TODO: account for start times being relative to different blocks!!!
            # check pause times make sense
            pauses=onsets[1:]-offsets[:-1]
            pause_tol=2 # pauses longer than this considered part of same segment
            #NOTE: maybe better to have separate parameter for pause tolerance and negative pause tolerance?
            if np.any(pauses<-pause_tol):
                print(f'there are {np.sum(pauses<-pause_tol)} bad pauses...')
                print('pruning stimuli around negative pauses based on relative confidence.')
                
                # gives array of indices corresponding to pauses
                # want to compare relative confidence values to decide if 
                # endpoints for stim before or after is kept 
                # -> compute relative confidence differential
                confidence_diff=np.diff(confidence)
                conf_diff_tol=0.1 # if differential is lower than this, throw away both stimuli before and after pause
                # if pause is bad AND confidence_diff +
                #  -> stim AFTER pause is more trustworthy
                # -> remove stim onset/offset BEFORE pause if bad and +
                bad_prepause_indx=np.flatnonzero(((pauses<-pause_tol)&(confidence_diff>0)|
                                                  ((pauses<-pause_tol)&(np.abs(confidence_diff)<conf_diff_tol))))
                # if pause is bad and confidence_diff - 
                # -> stim BEFORE pause is more trustworthy
                # -> remove stim on/off AFTER pause if bad and -
                bad_pstpause_indx=np.flatnonzero(((pauses<-pause_tol)&(confidence_diff<0)|
                                                  ((pauses<-pause_tol)&(np.abs(confidence_diff)<conf_diff_tol))))


                #check for distinct index values
                #NOTE COMMENTING OUT BC now that relative confidence is a factor, pre/post won't necessarily be unique if low differential
                # if utils.check_distinct_vals(bad_prepause_indx,bad_pstpause_indx):
                #     pass
                # else:
                #     raise NotImplementedError(f'indices should be different in bad_(pre,pst)_pause_indx arrays')
                # # make sure no confidence ties (very unlikely)
                # if len(np.flatnonzero((pauses<-pause_tol)&(confidence_diff==0)))>0:
                #     raise NotImplementedError( f"there are {len(np.flatnonzero((pauses<-pause_tol)&(confidence_diff==0)))} bad pauses with confidence ties.")
                # add one to post_pause, pretty sure this has to be done after uniqueness check, but resuling masks
                # should still index unique onsets/offsets
                bad_pstpause_indx+=1
                #NOTE: putting this code after adding one to post-pauses because for case where
                # confidence differential is low, decided to throw away both bordering stimuli
                
                # check for uniqueness as each index should be unique to individual stimuli
                if utils.check_distinct_vals(bad_prepause_indx,bad_pstpause_indx):
                    pass
                else:
                    raise NotImplementedError(f'Indices were unique bad_(pre,pst)_pause_indx arrays before adding one to pst, but now theyre not. and thats bad.')                 
                # prune the bad pauses as advertised, then re-evaluate pauses for segmenting
                mask=np.ones(onsets.shape).astype(bool)
                mask[np.concatenate([bad_prepause_indx,bad_pstpause_indx])]=False
                print(f"trimming {(~mask).sum()} stimuli with bad pause times.")

                onsets=onsets[mask]
                offsets=offsets[mask]
                
                pauses=onsets[1:]-offsets[:-1]
                if np.any(pauses<-pause_tol):
                    raise NotImplementedError("we got a problem here.")
                else:
                    print('negative pauses removed.')
                del mask
            else:
                print("no negative pauses to remove.")


            print("begin segmenting based on pauses.")
            long_pauses=np.flatnonzero(pauses>pause_tol)
            segment_offsets=offsets[long_pauses]
            segment_onsets=onsets[long_pauses+1]
            # correct for missing first onset, last offset
            segment_offsets=np.concatenate([segment_offsets,offsets[-1][None]])
            segment_onsets=np.concatenate([onsets[0][None],segment_onsets])


            #group stimuli by segments
            prev_stim_end=0
            for ii, (seg_start,seg_end) in enumerate(zip(segment_onsets,segment_offsets)):
                # seg_start,seg_end correspond to audio_rec samples
                rec_seg=audio_rec[seg_start:seg_end]
                t_rec=np.arange(seg_start,seg_end)/fs_eeg
                seg_idx=np.flatnonzero((seg_start<=onsets)&(onsets<seg_end))
                nms_in_seg=stim_nms[seg_idx]
                stims_in_seg=[]
                
                seg_start_time=seg_start/fs_eeg
                fig,ax=plt.subplots(2,sharex=True)
                ax[0].plot(t_rec,rec_seg/np.abs(rec_seg).max(),label=f"segment {ii+1}")
                ax[0].set_title(f"{subj_num} {block} segment {ii+1}")
                prev_stim_end=seg_start_time
                for stim_nm in nms_in_seg:
                    #note grabs clean by default
                    stim_wav=utils.get_stim_wav(stims_dict,stim_nm)
                    stims_in_seg.append(stim_wav)
                    t_stim=np.arange(stim_wav.size)/fs_audio
                    t_stim+=prev_stim_end
                    ax[1].plot(t_stim,stim_wav,label=f'{stim_nm[:-4]}')
                    prev_stim_end=t_stim[-1]
                ax[1].set_xlabel('time, s')
                box0=ax[0].get_position()
                ax[0].set_position([box0.x0,box0.y0+box0.height*0.1,
                                    box0.width,box0.height*0.9])
                box1=ax[1].get_position()
                ax[1].set_position([box1.x0,box1.y0+box1.height*0.1,
                                    box1.width,box1.height*0.9])
                ax[1].legend(loc='upper center',bbox_to_anchor=(0.5,-0.05),
                             ncol=len(nms_in_seg),
                             fontsize=9) #NOTE: can move using anchor box or some shit to make more readbale
                plt.show()
                    

                raise NotImplementedError("stop here.")
                

                
                
                
                
                
                
                #TODO: code to string together stimuli og wavs and eeg audio chnl recording
                # pertaining to the same continuous sound segment after pruning using logical or 
                # and both pad pause indexes immediately above this (maybe functionalize the code also?)
                #expecing about 15 per block x 6 blocks -> ~ 90 long pauses/segments





                break
            break





            

    else:
        raise NotImplementedError(f"{which_stmps} is not an option for which_stmps.")
 
  
# %%

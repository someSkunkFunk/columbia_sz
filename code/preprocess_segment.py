# Already have stim timestamps for each stim in some subjects; for those
# we just preprocess and segment
# for subjects without timestamps, first get the timestamps, then preprocess and segment
#%%

# INIT
import pickle
import scipy.io as spio
import numpy as np
import os
# import glob
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
n_blocks=6
# blocks=["B2"]
blocks = [f"B{ii}" for ii in range(1, n_blocks+1)]
print(f"check blocks: {blocks}")
def my_downsample(raw_eeg,filt_params,fs_in,fs_out):
    '''
    does anti-alias filter then downsamples using scipy's decimate
    ASSUMES BANDPASS
    ASSUMES TOTAL DOWNSAMPLING FACTOR OF 24
    RETURNS
    DECIMATED EEG
    '''
    filt_o=filt_params['order']
    filt_band_lims=filt_params['band_lims']
    down_factor1=6 # note that conveniently already integer valued in this case
    down_factor2=4
    fs_intermediate=int(fs_in/down_factor1)
    if down_factor1*down_factor2 != int(fs_in/fs_out):
        raise NotImplementedError(f"total downsampling ratio assumed to be 24, but here was: {int(fs_in/fs_out)}.")
    #NOTE: scipy docs recommends calling decimate multiple times when downsampling factor is higher than 13
    # I don't see why filtering would be required at decimations but not entirely sure how to bypass or make a unity filter
    
    b,a=signal.butter(filt_o,filt_band_lims,btype='bandpass',
                              output='ba',fs=fs_in)
    ftype=signal.dlti(b,a)
    decimated_eeg1=signal.decimate(raw_eeg,down_factor1,ftype=ftype,axis=0)
    del b,a,ftype
    # make new filter for second decimation at intermediate fs
    b,a=signal.butter(filt_o,filt_band_lims,btype='bandpass',
                              output='ba',fs=fs_intermediate)
    ftype=signal.dlti(b,a)
    decimated_eeg_out=signal.decimate(decimated_eeg1,down_factor2,ftype=ftype,axis=0)
    return decimated_eeg_out
#
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
    evnt_ovrall_thresh=float(os.environ["evnt_thresh"])
    print(f"evnt_ovrall_thresh is {evnt_ovrall_thresh, type(evnt_ovrall_thresh)}")
#####################################################################################
#manual vars
#####################################################################################
else:
    print("using manually inputted vars")
    subj_num="3283"
    which_stmps="evnt"
    which_xcorr="envs"
    just_stmp=False
    do_avg_ref=False
    evnt_ovrall_thresh=0.75
    noisy_or_clean="noisy" #NOTE: clean is default and setting them here does nothing
    
##################################################################################
### eeg preprocessing params
# NOTE: different from filter lims used in timestamp detection algo (!)
 #NEW: make bash var?
filt_params={
    "band_lims":[1.0, 15],
    "order":1
 } # order of filter (effective order x4 of this since using zero-phase and calling it twice)
#directory where processed data goes; 
# NOTE: we changed from specifying which stamps before
processed_dir_path=os.path.join(eeg_dir, "preprocessed_decimate") 
subj_cat=utils.get_subj_cat(subj_num) #note: checked get_subj_cat, should be fine
raw_dir=os.path.join(eeg_dir,"raw")
print(f"Fetching data for {subj_num,subj_cat}")
subj_eeg=utils.get_full_raw_eeg(raw_dir,subj_cat,subj_num,blocks=blocks)


#%%
# EXEC FIND TIMESTAMPS AND PREPROCESS
#print  correlation thresholds
timestamps_bad=True #currently unsure where the problem is but could still be timestamps although they seem good
# determine filter params applied to EEG before segmentation 

if which_stmps=="xcorr":
    # Find timestamps using xcorr algo
    thresh_params=('pearsonr', 0.4)
    # cutoff_ratio=10
    if thresh_params[0]=='xcorr_peak':
        print(f'using xcorr peak thresholding; xcorr cutoff: {thresh_params[1]} * std(xcorr)')
    elif thresh_params[0]=='pearsonr':
        print(f"using pearsonr thresholding; pearsonr threshold: {thresh_params[1]}")

    
    # check if save directory exists, else make one
    output_dir=os.path.join(processed_dir_path,subj_cat,subj_num)
    if not os.path.isdir(output_dir):
        print(f"preprocessed dir for {subj_num, subj_cat} not found, creating one.")
        os.makedirs(output_dir, exist_ok=True)
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
    thresh_dir=f"thresh_{round(evnt_ovrall_thresh*1000):03}"
    output_dir = os.path.join(processed_dir_path,thresh_dir,subj_cat,subj_num)
    if not os.path.isdir(output_dir):
        print(f"preprocessed dir for {subj_num, subj_cat} not found, creating one.")
        os.makedirs(output_dir, exist_ok=True)
    # find evnt timestamps
    evnt=utils.load_evnt(subj_num)
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
            # sos=signal.butter(filt_o,filt_band_lims,btype='bandpass',
            #                 output='sos',fs=fs_eeg)
            # filt_eeg=signal.sosfiltfilt(sos,raw_eeg,axis=0)
            audio_rec=raw_data[:,-1] # leave unperturbed for alignment checking
            # get number of samples in downsampled waveform
            # num_ds=int(np.floor((filt_eeg.shape[0]-1)*(fs_trf/fs_eeg)))
            # downsample eeg
            # NOTE: audio_rec not downsampled
            # subj_eeg[block]=(signal.resample(filt_eeg,num_ds,axis=0),audio_rec)
            # design decimate filter (bandpass in this case)

            # b,a=signal.butter(filt_o,filt_band_lims,btype='bandpass',
            #                   output='ba',fs=fs_eeg)
            # ftype=signal.dlti(b,a)
            # #NOTE: scipy docs recommends calling decimate multiple times when downsampling factor is higher than 13
            # #TODO: fix that ^
            # down_factor=int(fs_eeg/fs_trf) # note that conveniently already integer valued in this case
            # decimated_eeg=signal.decimate(raw_eeg,down_factor,ftype=ftype,axis=0)
            decimated_eeg=my_downsample(raw_eeg,filt_params,fs_in=fs_eeg,fs_out=fs_trf)
            subj_eeg[block]=(decimated_eeg,audio_rec)
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
        print(f'saving to: {output_dir}')
        subj_data.to_pickle(os.path.join(output_dir, f"{which_xcorr}_aligned_resp.pkl"))
        print(f"{subj_num, subj_cat} preprocessing and segmentation complete.")
        # break
    elif which_stmps.lower()=='evnt':
        print(f"using evnt timestamps to segment eeg.")
        evnt_nms,evnt_blocks,evnt_confidence,evnt_onsets,evnt_offsets=utils.extract_evnt_data(evnt,fs_eeg)
        # raise NotImplementedError("extract evnt info here.")
        # split into blocks 
        subj_output={} # save results in a python dictionary
        for block, raw_data in subj_eeg.items():
            print(f"segmenting block {block}...")
            block_idx=[eb.replace('0','')==block for eb in evnt_blocks]
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
            print("filtering and downsampling eeg for current block")
            # sos=signal.butter(filt_o,filt_band_lims,btype='bandpass',
            #                 output='sos',fs=fs_eeg)
            # filt_eeg=signal.sosfiltfilt(sos,raw_eeg,axis=0)
            # num_ds=int(np.floor((filt_eeg.shape[0]-1)*(fs_trf/fs_eeg)))
            # ds_eeg=signal.resample(filt_eeg,num_ds)
            ds_eeg=my_downsample(raw_eeg,filt_params,fs_in=fs_eeg,fs_out=fs_trf)
            
            # BELOW THIS LINE, CHANGE EVNT REFERENCES TO BLOCK REFERENCES
            # prune low-confidence values,record figure for posterity:
            
            print(f"Overall confidence threshold used: {evnt_ovrall_thresh}.")
            import matplotlib.pyplot as plt
            plt.hist(confidence,bins=25)
            plt.title(f"{subj_num}, {block} Evnt confidence vals")
            plt.axvline(evnt_ovrall_thresh,label=f'confidence threshold: {evnt_ovrall_thresh}')
            if "subj_num" in os.environ:
                #TODO: make this depend on the actual corrections
                corrections_dir='first_onset_correction'
                figs_dir=os.path.join("..","figures","evnt_info",thresh_dir,corrections_dir,subj_num,block)
                # delete old figures so new figures don't get confused with the old:
                #NOTE: CAREFUL W FOLLOWING FUNCTION SINCE DELETES FILES IN SUBDIRECTORIES OF FIGS_DIR
                utils.rm_old_figs(figs_dir)
                fig_pth=os.path.join(figs_dir, f"{subj_num}_{block}_confidence_hist.png")
                if not os.path.isdir(os.path.dirname(fig_pth)):
                    print(f"Making new figures directory: {os.path.dirname(fig_pth)}")
                    os.makedirs(os.path.dirname(fig_pth),exist_ok=True)
                plt.savefig(fig_pth)
                del fig_pth
            else:
                plt.show()
            plt.close()
            high_confidence=np.flatnonzero(confidence>evnt_ovrall_thresh)
            print(f"trimming {confidence.size-high_confidence.size} low confidence stims out of {confidence.size}.")
            onsets=onsets[high_confidence]
            offsets=offsets[high_confidence]
            confidence=confidence[high_confidence]
            stim_nms=stim_nms[high_confidence]
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
                if not utils.check_distinct_vals(bad_prepause_indx,bad_pstpause_indx):
                    raise NotImplementedError(f'Indices were unique bad_(pre,pst)_pause_indx arrays before adding one to pst, but now theyre not. and thats bad.')                 
                # prune the bad pauses as advertised, then re-evaluate pauses for segmenting
                mask=np.ones(onsets.shape).astype(bool)
                mask[np.concatenate([bad_prepause_indx,bad_pstpause_indx])]=False
                print(f"trimming {(~mask).sum()} stimuli with bad pause times.")

                onsets=onsets[mask]
                offsets=offsets[mask]
                stim_nms=stim_nms[mask]
                
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
            for seg_ii, (seg_start,seg_end) in enumerate(zip(segment_onsets,segment_offsets)):
                
                seg_nm=f"{block}_{seg_ii+1:02}"
                print(f"getting eeg from segment {seg_ii+1} of {len(segment_onsets)}")
                # seg_start,seg_end correspond to audio_rec samples
                rec_seg=audio_rec[seg_start:seg_end]
                t_rec=np.arange(seg_start,seg_end)/fs_eeg
                seg_idx=np.flatnonzero((seg_start<=onsets)&(onsets<seg_end))
                nms_in_seg=stim_nms[seg_idx]

                
                seg_start_time=seg_start/fs_eeg

                #plotting

                
                # Calculate the legend size
                fig_legend, ax_legend = plt.subplots()
                #NOTE: I think this is where the no artists with labels found warning is originating from 
                # number of columns in legend, need at least 1
                _ncol=max(len(nms_in_seg)//2, 1)
                ax_legend.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=_ncol)
                legend=ax_legend.get_legend()
                legend_box = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())
                fig_legend.clear()
                plt.close()
                fig_width=8
                fig_height=6
                
                # plot the data
                fig,ax=plt.subplots(2,sharex=True,figsize=(fig_width,fig_height))
                norm_aud_rec=rec_seg/np.abs(rec_seg).max()
                ax[0].plot(t_rec,norm_aud_rec,label=f"segment {seg_ii+1}")
                ax[0].set_title(f"{subj_num} {block} segment {seg_ii+1:02}")
                prev_stim_end=seg_start_time
                for stim_nm in nms_in_seg:
                    #note grabs clean by default
                    stim_wav=utils.get_stim_wav(stims_dict,stim_nm)
                    # stims_in_seg.append(stim_nm)
                    t_stim=np.arange(stim_wav.size)/fs_audio
                    t_stim+=prev_stim_end
                    ax[1].plot(t_stim,stim_wav,label=f'{stim_nm[:-4]}')
                    prev_stim_end=t_stim[-1]
                ax[0].set_xlabel('time, s, relative to block start')
                box0=ax[0].get_position()
                ax[0].set_position([box0.x0,box0.y0+box0.height*0.2,
                                    box0.width,box0.height*0.8])
                box1=ax[1].get_position()
                ax[1].set_position([box1.x0,box1.y0+box1.height*0.2,
                                    box1.width,box1.height*0.8])
                # adjust figure with legend outside of axes
                fig.subplots_adjust(bottom=legend_box.ymin * fig_height / (fig_height - legend_box.height))
                ax[1].legend(loc='upper center',bbox_to_anchor=(0.5,-0.1),
                             ncol=_ncol) 
                if "subj_num" in os.environ:
                    fig_pth=os.path.join(figs_dir,f"{subj_num}_{block}_{round(seg_ii+1):02}.png")
                    plt.savefig(fig_pth)
                    del fig_pth
                else:
                    plt.show()
                plt.close()

                # get downsampled segment onsets and offsets to slice downsampled eeg
                seg_start_ds=int((seg_start/fs_eeg)*fs_trf)
                seg_end_ds=int((seg_end/fs_eeg)*fs_trf)
                eeg_seg=ds_eeg[seg_start_ds:seg_end_ds]
                
                # record segmented eeg and stimuli names to a dictionary            
                subj_output[f"{seg_nm}"]=([n for n in nms_in_seg], 
                                          norm_aud_rec, eeg_seg)
                print(f"segment {seg_ii+1} of {len(segment_onsets)} done.")
        print(f"All segments done. saving preprocessed data to: {output_dir}")
        output_fnm=os.path.join(output_dir,f"aligned_resp.pkl")
        with open(output_fnm, 'wb') as fl:
            pickle.dump(subj_output,fl)
        print(f"Preprocessing and alignment done for {subj_num}.")
        #expecing about 15 per block x 6 blocks -> ~ 90 long pauses/segments
    else:
        raise NotImplementedError(f"{which_stmps} is not an option for which_stmps.")
 
  
# %%

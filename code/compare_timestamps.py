#%%
import utils
import scipy.io as spio
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import pearsonr
#%%
#select subject
do_all_subjs=True
stims_pth=os.path.join("..", "eeg_data", "stim_info.mat")
stims_dict=utils.get_stims_dict(stims_pth)
raw_dir=os.path.join("..", "eeg_data", "raw")
fs_eeg=2400
fs_audio=stims_dict['fs'][0]
# i suspect evnt timestamps are off by some constant
shift=12000

hist_ylims=[0,400]
if do_all_subjs:
    hcs=os.listdir(os.path.join(raw_dir, "hc"))
    sps=os.listdir(os.path.join(raw_dir, "sp"))
    all_subj_nums=hcs+sps
    for subj_num in all_subj_nums:
        print(f"subj_num={subj_num}:\n")
        # get test subject raw data
        subj_cat=utils.get_subj_cat(subj_num)
        subj_eeg=utils.get_full_raw_eeg(raw_dir,subj_cat,subj_num)
        
        
        # segment eeg using both timestamps

        # load my timestamps
        tmstmps_pth=os.path.join("..","eeg_data","timestamps",subj_cat,subj_num,"timestamps.pkl")
        with open(tmstmps_pth, 'rb') as f:
            timestamps=pickle.load(f)
        # flatten blocks so we can match flattened index
        all_nms=[]
        all_stmps=[]
        for _, nms_n_stamps in timestamps.items():
            for nm, stmps in nms_n_stamps.items():
                all_nms.append(nm)
                all_stmps.append(stmps)

        #load timestamps provided in evnt structure
        # NOTE: not sure if names will certainly be in order
        evnt_path=os.path.join("..", "eeg_data", "timestamps", f"evnt_{subj_num}.mat" )
        evnt_mat=spio.loadmat(evnt_path)
        # returns dict for some reason, which mat2dict doesnt like
        evnt=evnt_mat['evnt']
        mystery=utils.mat2dict(evnt)
        
        # slice each stim, compute pearson r vals
        segment_corrs=[]
        for stim_index in range(len(all_stmps)):
            print(f"running stim {stim_index} of {len(all_stmps)}")
        # get using weird indexing because object ndarray
            stim_nm_evnt=mystery['name'][0, stim_index][0]
            block_evnt=stim_nm_evnt[:3].capitalize().replace('0','')
            sync_position=mystery['syncPosition'][0,stim_index][0,0]
            # evnt_confidence=mystery['confidence'][0,stim_index][0,0]
            
            stim_nm_stmps=all_nms[stim_index]
            block_stmps=stim_nm_stmps[:3].capitalize().replace('0','')
            sync_positions=all_stmps[stim_index]
            # my_confidence=sync_positions[-1]


            # get eeg segments and corresponding stim wav


            if stim_nm_stmps!=stim_nm_evnt:
                raise NotImplementedError('stim names dont match')
            else:
                stim_nm=stim_nm_stmps

            #below assumes names match
            stim_wav=utils.get_stim_wav(stims_dict,stim_nm_stmps,'clean')
            stim_dur=(stim_wav.size-1)/fs_audio
            # segment eeg audio, downsample stim_wav, get correlations

            eeg_wav_evnt=subj_eeg[block_evnt][shift+sync_position:shift+sync_position+int(stim_dur*fs_eeg+1),-1]
            eeg_wav_stmps=subj_eeg[block_stmps][sync_positions[0]:sync_positions[1],-1]
            from scipy import signal
            sos = signal.butter(8, fs_eeg/3, fs=fs_audio, output='sos')
            stim_wav = signal.sosfiltfilt(sos, stim_wav)
            # downsample envelope or wav to eeg fs

                # will "undersample" since sampling frequencies not perfect ratio
            stim_wav_ds = signal.resample(stim_wav, int(np.floor(stim_dur*fs_eeg)+1))
            if sync_positions[0] is not None:
                segment_corrs.append((stim_nm, 
                                    {'evnt_vs_mine':pearsonr(eeg_wav_evnt,eeg_wav_stmps).statistic,
                                    'og_ds_vs_evnt':pearsonr(eeg_wav_evnt,stim_wav_ds).statistic,
                                    'og_ds_vs_mine':pearsonr(eeg_wav_stmps,stim_wav_ds).statistic
                                    }))

        # plot distributions for all confidence values
        my_confidence_vals=[]
        for (_,_,my_conf) in all_stmps:
            my_confidence_vals.append(my_conf)
        evnt_confidence_vals=[]
        for evnt_conf in mystery['confidence'].squeeze():
            evnt_confidence_vals.append(evnt_conf[0][0])

        fig, axs= plt.subplots(4,1,sharex=True)
        axs[0].hist(my_confidence_vals)
        axs[0].set_title('mine')
        axs[0].set_ylim(hist_ylims)
        axs[1].hist(evnt_confidence_vals)
        axs[1].set_title('evnt')
        axs[1].set_ylim(hist_ylims)
        axs[2].hist([stim['og_ds_vs_mine'] for _, stim in segment_corrs])
        axs[2].set_title('OG DS vs MINE')
        # axs[2].set_ylim(hist_ylims)
        axs[3].hist([stim['og_ds_vs_evnt'] for _, stim in segment_corrs])
        axs[3].set_title('OG DS VS EVNT')
        plt.show()

else:
    # select a single subject
    subj_num="3318"
    # get test subject raw data
    #TODO: update evnt_path so it reflects test subject

    raw_dir=os.path.join("..", "eeg_data", "raw")
    subj_cat=utils.get_subj_cat(subj_num)
    subj_eeg=utils.get_full_raw_eeg(raw_dir,subj_cat,subj_num)

    #%%
    # segment eeg using both timestamps
    stim_index=1

    #use timestamps provided in evnt structure
    # NOTE: not sure if names will certainly be in order
    evnt_path=os.path.join("..", "eeg_data", "timestamps", f"evnt_{subj_num}.mat" )
    evnt_mat=spio.loadmat(evnt_path)
    # returns dict for some reason, which mat2dict doesnt like
    evnt=evnt_mat['evnt']
    mystery=utils.mat2dict(evnt)
    # get using weird indexing because object ndarray
    stim_nm_evnt=mystery['name'][0, stim_index][0]
    block_evnt=stim_nm_evnt[:3].capitalize().replace('0','')
    sync_position=mystery['syncPosition'][0,stim_index][0,0]
    evnt_confidence=mystery['confidence'][0,stim_index][0,0]
    # use my timestamps
    tmstmps_pth=os.path.join("..","eeg_data","timestamps",subj_cat,subj_num,"timestamps.pkl")
    with open(tmstmps_pth, 'rb') as f:
        timestamps=pickle.load(f)
    # flatten blocks so we can match flattened index
    all_nms=[]
    all_stmps=[]
    for _, nms_n_stamps in timestamps.items():
        for nm, stmps in nms_n_stamps.items():
            all_nms.append(nm)
            all_stmps.append(stmps)
    stim_nm_stmps=all_nms[stim_index]
    block_stmps=stim_nm_stmps[:3].capitalize().replace('0','')
    sync_positions=all_stmps[stim_index]
    my_confidence=sync_positions[-1]


    # get eeg segments and corresponding stim wav


    if stim_nm_stmps!=stim_nm_evnt:
        raise NotImplementedError('stim names dont match')

    #below assumes names match
    stim_wav=utils.get_stim_wav(stims_dict,stim_nm_stmps,'clean')
    stim_dur=(stim_wav.size-1)/fs_audio
    eeg_wav_evnt=subj_eeg[block_evnt][shift+sync_position:shift+sync_position+int(stim_dur*fs_eeg+1),-1]
    eeg_wav_stmps=subj_eeg[block_stmps][sync_positions[0]:sync_positions[1],-1]
    #%% 
    # plot    


    plt.figure()
    plt.plot(np.arange(0,stim_wav.size)/fs_audio, stim_wav)
    plt.title('Stim Wav')
    plt.show()
    plt.figure()
    plt.plot(np.arange(0,eeg_wav_stmps.size)/fs_eeg, eeg_wav_stmps)
    plt.title(f'my EEG WAV, confidence {my_confidence: .2f}')
    plt.show()
    plt.figure()
    plt.plot(np.arange(0,eeg_wav_evnt.size)/fs_eeg,eeg_wav_evnt)
    plt.title(f'EEG WAV evnt, confidence: {evnt_confidence: .2f}')
    #%%
    # plot distributions for all confidence values
    my_confidence_vals=[]
    for (_,_,my_conf) in all_stmps:
        my_confidence_vals.append(my_conf)
    evnt_confidence_vals=[]
    for evnt_conf in mystery['confidence'].squeeze():
        evnt_confidence_vals.append(evnt_conf[0][0])

    fig, axs= plt.subplots(2,1,sharex=True)
    axs[0].hist(my_confidence_vals)
    axs[0].set_title('mine')
    axs[0].set_ylim(hist_ylims)
    axs[1].hist(evnt_confidence_vals)
    axs[1].set_title('evnt')
    axs[1].set_ylim(hist_ylims)
    plt.show()

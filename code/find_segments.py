# python script to manually segment each subject's eeg recordings...
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

#####################################################################################
#manual vars
#####################################################################################
else:
    print("using manually inputted vars")
    subj_num="3253"
############################################################################################
subj_cat=utils.get_subj_cat(subj_num) #note: checked get_subj_cat, should be fine
raw_dir=os.path.join(eeg_dir,"raw")
print(f"Fetching data for {subj_num,subj_cat}")
subj_eeg=utils.get_full_raw_eeg(raw_dir,subj_cat,subj_num,blocks=blocks)
# get experiment audio recording envelope, after de-trending via hi ghpass above 1 Hz
for block_num in blocks:
    rec_wav = subj_eeg[block_num][:,-1]
    filt_wav=sos=signal.butter(3,1.0,fs=fs_eeg,btype='high',output='sos')
    filt_wav=signal.sosfiltfilt(sos,rec_wav,)
    filt_rec_env = np.abs(signal.hilbert(filt_wav))
    
    
    
    # get segments where sound happened:
    print(f"splitting sound recording into segments")
    segments, smooth_envelope=utils.get_segments(filt_rec_env,fs_eeg)
    n_segments=segments.shape[0]
    print(f"{n_segments} segments found.")
    #### DEBUG PLOTTING###
    print("saving segment figures.")
    subj_debug_dir=os.path.join("..","figures","segments",subj_num,block_num)
    if not os.path.isdir(subj_debug_dir):
        os.makedirs(subj_debug_dir,exist_ok=True)
    import matplotlib.pyplot as plt
    t=np.arange(smooth_envelope.size)/fs_eeg
    on_times=t[segments[:,0]]
    off_times=t[segments[:,1]]
    plt.plot(t,filt_wav/np.max(np.abs(filt_wav)),label='highpassed recording')
    plt.plot(t,smooth_envelope,label='smooth_env')
    plt.stem(on_times,np.ones(on_times.shape),label='onsets',linefmt='green')
    plt.stem(off_times,np.ones(off_times.shape),label='offsets',linefmt='red')
    plt.xlabel('seconds')
    plt.legend(loc='lower left')
    plt.title(f"{subj_num} {block_num}")
    plt.tight_layout()
    fig_fnm=f"{subj_num}_{block_num}_segments.png"
    fig_pth=os.path.join(subj_debug_dir,fig_fnm)
    plt.savefig(fig_pth)
    plt.close()
    

    for ii, (on,off) in enumerate(segments):
        print(f"plotting segment {ii+1} of {len(segments)}...")
        plt.figure()
        recording_bit=filt_wav[on:off]
        recording_bit/=np.abs(recording_bit).max()
        plt.plot(t[on:off],recording_bit,label='highpassed recording')
        plt.title(f"{subj_num} {block_num} segment {ii+1}")
        plt.plot(t[on:off],smooth_envelope[on:off],label='smooth_env')
        plt.legend(loc="lower left")
        plt.xlabel('seconds')
        plt.tight_layout()
        fig_fnm=f"{subj_num}_{block_num}_segment{ii+1:02d}.png"
        fig_pth=os.path.join(subj_debug_dir,fig_fnm)
        plt.savefig(fig_pth)
        plt.close()
    #### END DEBUG PLOTTING###TODO: MAKE INTO A FUNCTION>>
#%%
import scipy.io as spio
from .mat2dict import mat2dict

def get_stims_dict(stim_fnm):
    '''
    transform mat file to dictionary for 
    '''
    # NOTE: determined that both stim files have same orig_noisy stim wavs
    
    all_stims_mat = spio.loadmat(stim_fnm, squeeze_me=True)

    # NOTE: I think "orig_clean" and "orig_noisy" are regular wavs and 
    #   "aud_clean"/"aud_noisy" are 100-band spectrograms?
    # according to master_docEEG spectrograms are at 100 Hz? 
    all_stims = all_stims_mat['stim']
    # convert structured array to a dict based on dtype names 

    # (which correspond to matlab struct fields)
    stims_dict = mat2dict(all_stims)
    #NOTE: stim durations between 0.87s and 10.6 s
    return stims_dict

#%%
#NOTE: this test will fail but I think the function works still?
#  (when called from another script)
import os
if __name__ == '__main__':
    stims_fnm = os.path.join(os.getcwd(),"..",
                                "eeg_data",
                                'stims_dict.pkl')
    stims_dict = get_stims_dict()
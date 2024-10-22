#%%
import utils

#%%
#TODO: need to re-align stimuli envelopes with responses... 
# I think we can take previously aligned data and it should have the stimuli
#  names which we can use to match and swap the new features with....? 
#%%
thresh_dir=f"thresh_{evnt_thresh}" # should be three digit number representing decimals to third place
prep_data_dir=os.path.join(eeg_dir,'preprocessed_decimate',thresh_dir)
subj_data=utils.load_preprocessed(subj_num,eeg_dir=prep_data_dir,
                                    evnt=evnt,which_xcorr=which_xcorr)
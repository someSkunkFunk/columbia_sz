#%%
import os
import pickle
from .get_subj_cat import get_subj_cat
def load_subj_data(subj_num, eeg_dir=None):
    '''
    helper to load subject data as pandas dataframe (pkl fl)
    '''
    subj_cat = get_subj_cat(subj_num)
    if eeg_dir is None:
        eeg_dir = os.path.join(os.getcwd(), '..', "eeg_data")
    subj_data_fnm = subj_num+"_prepr_data.pkl"
    with open(os.path.join(eeg_dir, subj_cat, subj_num, subj_data_fnm), 'rb') as file:
        subj_data = pickle.load(file)
    return subj_data

#%%
# NOTE: this test will fail but I think when calling the module from
#  another script that is not within utils it will work?
#  seems to be the case for calling mat2dict within get_stims_dict
if __name__ == '__main__':
    test_subj = '3316'
    subj_data = load_subj_data(test_subj)
    subj_data.head()
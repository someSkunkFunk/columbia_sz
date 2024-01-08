import pickle
import os
def load_stims_dict():
    stims_fnm = os.path.join("..",
                                "eeg_data",
                                'stims_dict.pkl')
    with open(stims_fnm, 'rb') as file:
        stims_dict = pickle.load(file)
    return stims_dict
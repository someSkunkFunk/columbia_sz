#%%
import os
def get_subj_cat(subj_num, eeg_dir=None):
    """
    create dictionary for looking up subject number based on healthy control (hc) or 
    schizophrenia patient (sp) categories
    """
    if eeg_dir is None:
        # lookup based on folder structure of eeg_dir, default:
        eeg_dir = os.path.join( '..', "eeg_data")
    hc_folder = os.path.join(eeg_dir, "raw", "hc")
    sp_folder = os.path.join(eeg_dir, "raw","sp")

    # print("eeg_dir: ", eeg_dir)
    if os.path.exists(os.path.join(hc_folder, subj_num)):
        return "hc"
    elif os.path.exists(os.path.join(sp_folder, subj_num)):
        return "sp"
    else:
        raise NotImplementedError("subj category could not be found.")
    
#%%    
if __name__=="__main__":
    hc_test_subj = "3316"
    sp_test_subj = "2782"
    print("hc", get_subj_cat(hc_test_subj,
                       eeg_dir=os.path.join(os.getcwd(), "..", "..","eeg_data")
))
    print("sp", get_subj_cat(sp_test_subj,
                       eeg_dir=os.path.join(os.getcwd(), "..", "..","eeg_data")
))
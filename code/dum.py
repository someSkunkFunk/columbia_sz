# TODO: come up with better name for this
# basically a backward model analysis script that is less complex than our current og_env_recon.py now that we have less parameters to vary
#%%
import utils
import os
import scipy.io as spio
import numpy as np

stimuliFile='speechFeats.mat'
stimuliPath=os.path.join("..","stimuli",stimuliFile)
stimData=spio.loadmat(stimuliPath)
fsStim=stimData['fs']
speechFeats=stimData['speechFeats']
#%%
def matlab_struct_to_dict(mat_struct):
    """ 
    Recursively converts a MATLAB structure (from scipy's loadmat) to a 
    Python dictionary. 
    NOTE: maybe replace mat2dict in our utils
    """
    if isinstance(mat_struct, np.ndarray) and mat_struct.dtype.names is not None:
        # MATLAB structure as numpy array of objects
        return {name: matlab_struct_to_dict(mat_struct[name]) for name in mat_struct.dtype.names}
    elif isinstance(mat_struct, np.ndarray) and mat_struct.size > 0:
        # Array of values
        return [matlab_struct_to_dict(element) for element in mat_struct]
    else:
        # Base case: return the value (could be int, float, string, etc.)
        return mat_struct


speechFeats=matlab_struct_to_dict(speechFeats)
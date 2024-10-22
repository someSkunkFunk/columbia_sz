# TODO: come up with better name for this
# basically a backward model analysis script that is less complex than our current og_env_recon.py now that we have less parameters to vary
# NOTE: this might end up becoming just a stimuli formatting script 
#%%
import utils
import os
# import scipy.io as spio
import numpy as np
from mat4py import loadmat
import pickle as pkl

rectify=True
norm=True
noisyOrClean='clean'
stimuliFeatsFile='speechFeats.mat'
stimuliFeatsPath=os.path.join("..","stimuli",stimuliFeatsFile)
outFileName=f'{noisyOrClean}Envs.pkl'
outFilePath=os.path.join("..","stimuli",outFileName)
# stimData=spio.loadmat(stimuliPath)
# note: this function only uses native python types so arrays get unwrapped
# into nested lists so sgram is a list with outer dimension being time and inner
# dimension being critical bands
stimFeatsData=loadmat(stimuliFeatsPath)
fsStimFeats=stimFeatsData['fs']
#%%
stimEnvs=[np.asarray(x['env']) for x in stimFeatsData['speechFeats'][noisyOrClean]]
if rectify:
    # loop feels stupid here but i am stupid
    for idx,env in enumerate(stimEnvs):
        env[env<0]=0
        stimEnvs[idx]=env
    del idx,env

if norm:
    envStd=np.std(np.concatenate(stimEnvs),ddof=1)
    stimEnvs=[x/envStd for x in stimEnvs]

with open(outFilePath,'wb') as file:
    pkl.dump(
        {
            'envelopes':stimEnvs,
            'fs':fsStimFeats,
            'normalized':norm,
            'rectified':rectify

        }, file
    )
# now they're normalized... perhaps we want to put into stim info moving forward
# or perhaps keep separate in similar format as clean_rms_envelopes for backwards compatibility
# in any case, let's take a detour just to see what stim info looks like when imported 
# using our new function instead of old custom made functions
#%%
# stimInfoPath=os.path.join("..","stimuli","stim_info.mat")
# stimInfoData=loadmat(stimInfoPath) # doesn't work
# stimsDict=utils.get_stims_dict()



# speechFeats=stimData['speechFeats']
#%%
# def matlab_struct_to_dict(mat_struct):
#     """ 
#     Recursively converts a MATLAB structure (from scipy's loadmat) to a 
#     Python dictionary. 
#     NOTE: maybe replace mat2dict in our utils
#     """
#     if isinstance(mat_struct, np.ndarray) and mat_struct.dtype.names is not None:
#         # MATLAB structure as numpy array of objects
#         return {name: matlab_struct_to_dict(mat_struct[name]) for name in mat_struct.dtype.names}
#     elif isinstance(mat_struct, np.ndarray) and mat_struct.size > 0:
#         # Array of values
#         return [matlab_struct_to_dict(element) for element in mat_struct]
#     else:
#         # Base case: return the value (could be int, float, string, etc.)
#         return mat_struct


# speechFeats=matlab_struct_to_dict(stimFeatsData['speechFeats'])
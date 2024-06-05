# script for making visual guide of gtec eelctrode locations
#Script for plotting trf decoder weights; can do grand average across all subjects, within a single category, or single subject

#%%
# INIT

import numpy as np
import os
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
from utils import get_gtec_pos
def topo_wrapper(display_vals,
                 fig_axes:tuple,
                 title:str,
                 save_pth:str=None,
                 posarr='gtec62',
                 show=True,
                 vlims=(None,None)):
    '''
    fig_axes: (fig, axes) as tuple; make sure axes is a single axes object onto 
    which display_vals are to be plotted
    '''
    from mne.viz import plot_topomap
    fig,ax=fig_axes
    if posarr=='gtec62':
        posarr=get_gtec_pos()
    else:
        raise NotImplementedError("Just use default value for posarr.")
    im,_=plot_topomap(display_vals,posarr,axes=ax,show=False,vlim=vlims)
    ax.set_title(title)
    fig.colorbar(im,ax=ax)
    if save_pth is not None:
        save_dir=os.path.dirname(save_pth)
        print("saving fig...")
        if os.path.isdir(save_dir):
            plt.savefig(save_pth)
        else:
            os.makedirs(save_dir,exist_ok=True)
            plt.savefig(save_pth)         
    if show:
        plt.show()


#%%
# topoplotting script

if __name__=='__main__':
    
    n_electrodes=62
    for n_elec in range(n_electrodes):
        #TODO: check what happens with 1-D array (which we did in corr_aligned_eeg) 
        # vs 2-D array but singleton dim?
        disp_vals=np.zeros(n_electrodes)
        disp_vals[n_elec]+=1
        fig,ax=plt.subplots()
        save_pth=os.path.join("..","figures","topo_visguide",f"electrode_{n_elec+1}.png")
        topo_wrapper(display_vals=disp_vals,
                     fig_axes=(fig,ax),
                     title=f"electrode {n_elec+1}",
                     save_pth=save_pth)





# %%

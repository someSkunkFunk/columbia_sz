import matplotlib.pyplot as plt
import numpy as np
def plot_waveform(x:list, fs:list, labels=[None], **kwargs):
    '''
    wrapper around subplots to plot multiple waveforms
    NOTE: implicitly assumes nrows and ncols correspond w number of waveforms given
    which we may want to change
    fs: sampling frequency as float or list if each waveform sampled at different rate
    x: 1d array or list of arrays if multiple waveforms
    label: MUST be a list of strings (or None), one per x
    '''

    fig, axs = plt.subplots(**kwargs)
    #TODO: when plotting on same axes, sort waveforms so they are easy to look at
    if any(labels) and len(labels) != len(x):
        #TODO: try except might make more sense here but idk how to define exception properly still
        raise NotImplementedError(f'''{len(labels)} number of labels doesnt 
                                  match number of waveforms ({len(x)})''')

    if isinstance(axs, plt.Axes):
        # plot waveforms on same axes
        if isinstance(x, list) and isinstance(fs, list):
            print('a')
            # different sampling frequencies
            for xi, fsi, label in zip(x, fs, labels):
                t = make_time_vector(fsi, xi.size)
                axs.plot(t, xi, label=label)
        elif isinstance(x, list) and not isinstance(fs, list):
            print('b')
            # same sampling frequencies
            for xi, label in zip(x, labels):
                t = make_time_vector(fs, xi.size)
                axs.plot(t, xi, label=label)
        else:
            print('c')
            # single waveform
            t = make_time_vector(fs, x.size)
            axs.plot(t, x, label=labels)
    else:
        # plot waveforms on separate axes
        if isinstance(x, list) and isinstance(fs, list):
            # different sampling frequencies
            for xi, fsi, ax, label in zip(x, fs, axs.flatten(), labels):
                t = make_time_vector(fsi, xi.size)
                print(xi.shape)
                ax.plot(t, xi, label=label)
                ax.legend()
        elif isinstance(x, list) and not isinstance(fs, list):
            # same sampling frequencies
            for xi, ax, label in zip(x, axs.flatten(), labels):
                t = make_time_vector(fs, xi.size)
                ax.plot(t, xi, label=label)
                ax.legend()
        else:
            raise NotImplementedError(f'x needs to be a list of waveforms when multiple axes.')

        
    # plt.tight_layout()
    # plt.show()
    return fig, axs
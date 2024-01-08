import numpy as np
def make_time_vector(fs, nsamples, start_time=0):
    return np.arange(start_time, nsamples/fs, 1/fs)
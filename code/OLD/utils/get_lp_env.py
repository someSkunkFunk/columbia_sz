from scipy import signal
def get_lp_env(env, crit_freq, fs):
    '''
    env: recording envelope
    crit_freq: lowpass filter cutoff f
    fs: recording fs
    '''
    # create lowpass filter
    sos = signal.butter(16, crit_freq, fs=fs, output='sos')
    lp_env = signal.sosfiltfilt(sos, env)
    return lp_env

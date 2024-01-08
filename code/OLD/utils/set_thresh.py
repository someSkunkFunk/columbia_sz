import numpy as np
import matplotlib.pyplot as plt
def set_thresh(og_rec, lp_env, fs):
    
    t = np.arange(0, og_rec.size/fs, 1/fs)
    ylim = [0, og_rec.std()*3.0]
    plt.plot(t, og_rec, t, lp_env)
    plt.xlabel('Seconds')
    plt.ylim(ylim)
    plt.show()
    satisfaction = ['y', 'ya', 'sure', 'yes', 'ok', 'why not']
    satisfied = False
    thresh = None
    feelings=None
    while not satisfied:
        thresh = float(input('Choose a threshold.'))
        on_off_times = np.zeros(lp_env.size, dtype=bool)
        on_off_times[1:] = np.diff(lp_env>thresh)
        plt.plot(t, og_rec, t, lp_env)
        plt.xlabel('Seconds')
        plt.hlines(thresh, t[0], t[-1], color='r', label='thresh')
        plt.vlines(t[on_off_times], ylim[0], ylim[1], color='r')
        plt.ylim(ylim)
        plt.show()
        feelings = input('Threshold good enough? y/n').lower()



        
        if feelings in satisfaction:
            garbage_out = False
            while not garbage_out:
                # endpoints might be stupid
                first = ['f', 'first', '1']
                last = ['l', 'last', '2']
                #eg: "first and last" gets rid of both
                discard_endpoints = input('Discard endpoints? (first and/or last)').lower()
                if any([(f in discard_endpoints) for f in first]):
                    # ignore first n thresh crossing
                    #NOTE: must be an int
                    n_bad_starts = int(input('How many false starts? (int)'))
                    on_off_times[np.where(on_off_times)[0][np.arange(n_bad_starts)]] = 0             
                if any([(l in discard_endpoints) for l in last]):
                    # ignore last n thresh crossing
                    n_bad_ends = int(input('How many false starts? (int)'))
                    on_off_times[np.where(on_off_times)[0][-np.arange(n_bad_ends)]] = 0
                # visual confirmation:
                plt.plot(t, og_rec, t, lp_env)
                plt.xlabel('Seconds')
                plt.hlines(thresh, t[0], t[-1], color='r', label='thresh')
                plt.vlines(t[on_off_times], ylim[0], ylim[1], color='r')
                plt.ylim(ylim)
                plt.show()
                feelings = input('All bad crossings out?').lower()
                if feelings in satisfaction:
                    garbage_out = True
                    satisfied = True

    #NOTE: on_off_times is actually bool array with ones where lp envelope crosses
    # threshold, not actually a list of times
    return thresh, on_off_times
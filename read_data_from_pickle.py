import numpy as np
import pickle

def read_from_pickle(path, time_steps = 5):
    f = open(path,'rb')
    x = pickle.load(f)
    f.close()

    keys = ['x','y','z','doppler', 'peakVal']
    for time_segment in range(len(x)//time_steps):
        for t in range(time_steps):
            s = np.array([x[time_segment*time_steps+t].get(k).mean() for k in keys])
            if t == 0:
                sample = s
            else:
                sample = np.vstack((sample , s))

        sample = np.expand_dims(sample, axis=0)
        if time_segment == 0:
            data = sample
        else:
            data = np.vstack((data, sample))
    return data
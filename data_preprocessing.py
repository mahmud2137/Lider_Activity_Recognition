import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def reading_sensor_values(path):
    started = False
    sensor_values = []
    intensity_values = []
    with open(path, "r") as fp:
        for i, line in enumerate(fp.readlines()):
            if line.rstrip() == "[scan]": 
                started = True
                continue
            if started and line.rstrip()=="[timestamp]":
                started = False
                continue
            # process line 
            if started:
                numbers = re.findall(f'\d+', line) 
                distances = numbers[::2]
                intensities = numbers[1::2]
                distances = [int(x) for x in distances]
                intensities = [int(x) for x in intensities]
                distances.extend(intensities)
                sensor_values.append(distances)
                # intensity_values.append(intensities)

    return np.array(sensor_values)#, np.array(intensity_values)

def downsample_n_segment(data, down_sample_factor=2, segment_size = 40):
    data = data[::down_sample_factor] #downsample to 20 frames per second
    n_segments = data.shape[0]//segment_size
    data = data[0:n_segments*segment_size]
    data_segmented = data.reshape(-1,segment_size,data.shape[1])

    return data_segmented

if __name__ == "__main__":
    walking = reading_sensor_values('walking.txt')
    walking_segmented = downsample_n_segment(walking)
    print(walking_segmented.shape)
    



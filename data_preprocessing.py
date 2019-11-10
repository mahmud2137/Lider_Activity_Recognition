import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def reading_sensor_values(path):
    started = False
    sensor_values = []
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
                values = re.findall(f'\d+', line)[::2]
                values = [int(x) for x in values]
                sensor_values.append(values)

    return np.array(sensor_values)

sv = reading_sensor_values('walking.txt')
print(sv.shape)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


def read_data(data):
    data = data.groupby('Frame #').mean()
    data.drop(columns = ['# Obj'], inplace = True)
    data.reset_index(drop= True, inplace = True)
    return data




def segment_n_reshape(data, n_time_step = 10 ):

    n_sample = data.shape[0]//n_time_step

    data = data.iloc[0:n_sample*n_time_step].values
    data = data.reshape((n_sample, n_time_step, -1))
    return data


if __name__=='__main__':
    data_dir = 'data/'
    # inactive_data = pd.read_csv(data_dir + 'inactive_one.csv')
    # inactive_data.groupby('Frame #').mean()

    walking_data = pd.read_csv(data_dir + 'walk_one.csv')
    walking_data = read_data(walking_data)
    walking_data = segment_n_reshape(walking_data)

    inactive_data = pd.read_csv(data_dir + 'inactive_1.csv')
    inactive_data = read_data(inactive_data)
    inactive_data
    
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Softmax, Dropout, Conv1D, TimeDistributed, MaxPooling1D, Flatten
# from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from data_preprocessing import *

# Data reading and preparing
walking = reading_sensor_values('walking.txt')
seating  = reading_sensor_values(('seating.txt'))

walking = downsample_n_segment(walking)
seating = downsample_n_segment(seating)
X = np.append(walking,seating, axis=0)
y = np.append(np.zeros(walking.shape[0]),np.ones(seating.shape[0])) # 0 is for walking and 1 is for seating

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
X_train.shape

scalers = {}
for i in range(X_train.shape[2]):
    scalers[i] = MinMaxScaler()
    X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i]) 

for i in range(X_test.shape[2]):
    X_test[:, :, i] = scalers[i].transform(X_test[:, :, i]) 

# print(X_train[0])
# print(X)

input_shape = X_train.shape[1:3]
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=5, kernel_size=3, activation='relu'),input_shape=input_shape)
model.add(TimeDistributed(MaxPooling1D(pool_size = 2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100, stateful = True, return_sequences=True))
model.add(LSTM(50, stateful = True))
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Softmax())
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

hist = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)
y_pred = model.predict(X_train)
test_acc = accuracy_score(y_test, y_pred)
print(test_acc)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Softmax, Dropout
from keras.optimizers import Adam
# from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from data_pp_mmwave import *


data_dir = 'data/'


walking_data = pd.read_csv(data_dir + 'walk_1.csv')
walking_data = read_data(walking_data)
walking_data = segment_n_reshape(walking_data)
data = walking_data
label = np.ones(walking_data.shape[0])  # Label is 1 for walking and 0 for inactivity

walking_data = pd.read_csv(data_dir + 'walk_2.csv')
walking_data = read_data(walking_data)
walking_data = segment_n_reshape(walking_data)

data = np.vstack((data,walking_data))
label = np.append(label,np.ones(walking_data.shape[0]))

for i in [1,2,3]:
    inactive_data = pd.read_csv(data_dir + f'inactive_{i}.csv')
    inactive_data = read_data(inactive_data)
    inactive_data = segment_n_reshape(inactive_data)

    data = np.vstack((data,inactive_data))
    label = np.append(label,np.zeros(inactive_data.shape[0]))


X_train, X_test, y_train, y_test = train_test_split(data,label, test_size=0.2)

scalers = {}
for i in range(X_train.shape[2]):
    scalers[i] = StandardScaler()
    X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i]) 

for i in range(X_test.shape[2]):
    X_test[:, :, i] = scalers[i].transform(X_test[:, :, i]) 



opt  = Adam(learning_rate=0.0001)
input_shape = X_train.shape[1:3]
model = Sequential()
model.add(LSTM(20, input_shape=input_shape, return_sequences=False))
# model.add(LSTM(100))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.5))
# model.add(Dense(50, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer=opt)
model.summary()
hist = model.fit(X_train, y_train, epochs=400,
                validation_split=0.1,
                batch_size=32, verbose=2)

y_pred = model.predict(X_test)
y_pred  = y_pred > 0.5
y_pred = y_pred.astype('int').reshape(-1)
test_score = sum(y_test == y_pred)/len(y_test)
print('test score:',test_score)

plt.figure(figsize=(10,8))
plt.plot(hist.history['loss'], label = 'Training Loss')
plt.plot(hist.history['val_loss'], label = 'Validation Loss')
plt.legend(loc = 'best')
plt.savefig('learning_curve.png')


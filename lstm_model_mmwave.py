import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Softmax, Dropout
from keras.callbacks.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
# from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import matplotlib.pyplot as plt
import pandas as pd
from data_pp_mmwave import *
from read_data_from_pickle import *
import os

data_dir = 'data/'


# walking_data = pd.read_csv(data_dir + 'walk_1.csv')
# walking_data = read_data(walking_data)
# walking_data = segment_n_reshape(walking_data)
# data = walking_data
# label = np.ones(walking_data.shape[0])  # Label is 1 for walking and 0 for inactivity

# walking_data = pd.read_csv(data_dir + 'walk_2.csv')
# walking_data = read_data(walking_data)
# walking_data = segment_n_reshape(walking_data)


data  = read_from_pickle(data_dir + 'data_sitting.pkl', 5)
label = np.zeros(data.shape[0])

data_standing = read_from_pickle(data_dir + 'data_standing.pkl', 5)
data = np.vstack((data, data_standing))
label = np.append(label,np.ones(data_standing.shape[0]))

data_walking = read_from_pickle(data_dir + 'data_walking.pkl', 5)
data = np.vstack((data, data_walking))
label = np.append(label, 2 * np.ones(data_walking.shape[0]))


for i in [1,2,3]:
    inactive_data = pd.read_csv(data_dir + f'inactive_{i}.csv')
    inactive_data = read_data(inactive_data)
    inactive_data = segment_n_reshape(inactive_data)

    data = np.vstack((data,inactive_data))
    label = np.append(label,3 * np.ones(inactive_data.shape[0]))

label = to_categorical(label)



X_train, X_test, y_train, y_test = train_test_split(data,label, test_size=0.2)

scalers = {}
for i in range(X_train.shape[2]):
    scalers[i] = StandardScaler()
    X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i]) 

for i in range(X_test.shape[2]):
    X_test[:, :, i] = scalers[i].transform(X_test[:, :, i]) 



opt  = Adam(lr=0.0001)
call_back = EarlyStopping(patience=5)
input_shape = X_train.shape[1:3]
model = Sequential()
model.add(LSTM(20, input_shape=input_shape, return_sequences=False))
# model.add(LSTM(100))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.5))
# model.add(Dense(50, activation = 'relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()
hist = model.fit(X_train, y_train, epochs=400,
                validation_split=0.1,
                callbacks= [call_back],
                batch_size=32, verbose=2)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

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




# pred = model.predict(sample)
# pred = 1 if pred>0.5 else 0




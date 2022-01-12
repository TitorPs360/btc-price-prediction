import numpy as np
# Please use numpy version 1.19 to compatible with LSTM tensor

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))

# read csv file
dataset = pd.read_csv('./BTC-USD.csv')

# delete useless column
dataset = dataset.drop(['Date', 'Adj Close'], axis=1)

# print head of dataset
print(dataset.head(15))

# split train 80% test 20%
n_train_rows = int(dataset.shape[0]*.8) - 1

# Split into train and test sets but keep all columns
train = dataset.iloc[:n_train_rows, :]
test = dataset.iloc[n_train_rows:, :]

# normalize data
train_set_scaled = sc.fit_transform(train.values)
test_set_scaled = sc.fit_transform(test.values)

# train dataset
X_train = []
Y_train = []

# test dataset
X_test = []
Y_test = []

# load data range
window = 50

# generate train dataset
for i in range(window, train_set_scaled.shape[0]):
    X_train.append(train_set_scaled[i - window: i, :])

    Y_train.append(train_set_scaled[i, :])

# generate test dataset
for i in range(window, test_set_scaled.shape[0]):
    X_test.append(test_set_scaled[i - window: i, :])

    Y_test.append(test_set_scaled[i, :])

# make list to np array
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_test, Y_test = np.array(X_test), np.array(Y_test)

# make model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True,
          input_shape=(X_train.shape[1], 5)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=5))

# compile model
model.compile(loss="mse", optimizer="adam")

# train model
model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_split=0.1)

# save model
model.save('LSTM_model.h5')

# eval model
print('\n# Evaluate on test data')
results = model.evaluate(X_test, Y_test, batch_size=32)
print('test loss, test acc:', results)

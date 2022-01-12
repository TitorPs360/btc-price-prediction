import numpy as np

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))

# read csv file
dataset = pd.read_csv('./BTC-USD.csv')

# get close price
train_set = dataset.iloc[:, 5:6].values

# normalize data
train_set_scaled = sc.fit_transform(train_set)

# train dataset
X_train = []
Y_train = []

# test dataset
X_test = []
Y_test = []

# test range
reject = 30

# load data range
window = 10

# generate train dataset
for i in range(0, len(train_set_scaled) - reject - window - 1):
    X_train.append(train_set_scaled[i: i + window, 0])

    if dataset.iloc[i + window + 1, 5] - dataset.iloc[i + window, 5] > 0:
        Y_train.append(1)
    else:
        Y_train.append(0)

# generate test dataset
for i in range(len(train_set_scaled) - reject - window, len(train_set_scaled) - window - 1):
    X_test.append(train_set_scaled[i: i + window, 0])

    if dataset.iloc[i + window + 1, 5] - dataset.iloc[i + window, 5] > 0:
        Y_test.append(1)
    else:
        Y_test.append(0)


# make list to np array
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_test, Y_test = np.array(X_test), np.array(Y_test)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

# make model
model = Sequential()
model.add(Dense(12, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# train model
model.fit(X_train, Y_train, epochs=1000, batch_size=10)

# eval model
_, accuracy = model.evaluate(X_test, Y_test)
print('\nAccuracy: %.2f' % (accuracy * 100))

# save model
model.save('classification_model.h5')

import numpy as np

import pandas as pd

from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))

# read csv file
dataset = pd.read_csv('./BTC-USD.csv')

# get close price
test_set = dataset.iloc[:, 5:6].values

# normalize data
test_set_scaled = sc.fit_transform(test_set)

# test dataset
X_test = []
Y_test = []

# test range
reject = 30

# load data range
window = 10

# generate test dataset
for i in range(0, len(test_set_scaled) - window - 1):
    X_test.append(test_set_scaled[i: i + window, 0])

    if dataset.iloc[i + window + 1, 5] - dataset.iloc[i + window, 5] > 0:
        Y_test.append(1)
    else:
        Y_test.append(0)


# make list to np array
X_test, Y_test = np.array(X_test), np.array(Y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

model = load_model('./classification_model.h5')

# eval model
_, accuracy = model.evaluate(X_test, Y_test)
print('\nAccuracy: %.2f' % (accuracy * 100))

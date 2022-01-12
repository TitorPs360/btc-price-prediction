import numpy as np
# Please use numpy version 1.19 to compatible with LSTM tensor

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler

sc_history = MinMaxScaler(feature_range=(0, 1))
sc_future = MinMaxScaler(feature_range=(0, 1))

# read csv file
dataset = pd.read_csv('./BTC-USD.csv')

# delete useless comlumn
dataset = dataset.drop(['Adj Close'], axis=1)

# split history 80% future 20%
n_history_rows = int(dataset.shape[0]*.95) - 1

# Split future and history sets but keep all columns
history = dataset.iloc[:n_history_rows, 1:]
future = dataset.iloc[n_history_rows:, 1:]

# define simulated today
today = dataset.iloc[n_history_rows, 0]

# normalize data
history_set_scaled = sc_history.fit_transform(history.values)
future_set_scaled = sc_future.fit_transform(future.values)

# history dataset
X_history = []
Y_history = []

# future dataset
X_future = []
Y_future = []

# load data range
window = 50

# generate history dataset
for i in range(window, history_set_scaled.shape[0]):
    X_history.append(history_set_scaled[i - window: i, :])

    Y_history.append(history_set_scaled[i, :])

# generate future dataset
for i in range(window, future_set_scaled.shape[0]):
    X_future.append(future_set_scaled[i - window: i, :])

    Y_future.append(future_set_scaled[i, :])

# make list to np array
X_history, Y_history = np.array(X_history), np.array(Y_history)
X_future, Y_future = np.array(X_future), np.array(Y_future)

# load model
model = load_model('./LSTM_model.h5')

# predict history data
Y_predict_history = model.predict(X_history)
Y_predict_history = sc_history.inverse_transform(Y_predict_history)
Y_real_history = history[window:].reset_index()

# predict future data
Y_predict_future = model.predict(X_future)
Y_predict_future = sc_future.inverse_transform(Y_predict_future)
Y_real_future = future[window:].reset_index()

# define history x label
all_date_history = dataset.iloc[window:n_history_rows, 0].values
x_index_history = list(range(len(all_date_history)))
date_label_history = []
x_tick_history = []

# define history x label
all_date_future = dataset.iloc[n_history_rows+window:, 0].values
x_index_future = list(range(len(all_date_future)))
date_label_future = []
x_tick_future = []

# define x trick to plot history
for i in range(0, len(all_date_history), 200):
    date = all_date_history[i]
    date_label_history.append(date)
    x_tick_history.append(i)

# define x trick to plot future
for i in range(0, len(all_date_future), 40):
    date = all_date_future[i]
    date_label_future.append(date)
    x_tick_future.append(i)

# visualise the history real close price
plt.figure(figsize=(18, 9))
plt.plot(x_index_history, Y_real_history['Close'],
         color='red', label='Real Close Price')
plt.xticks(x_tick_history, date_label_history)
plt.title('Real Close Price vs Predict Close Price [History]')
plt.ylabel('Close Price')
plt.xlabel(f'Date [Simulated Today is {today}]')
plt.legend()
plt.show()

# visualise the history prediction close price
plt.figure(figsize=(18, 9))
plt.plot(x_index_history, Y_real_history['Close'],
         color='red', label='Real Close Price')
plt.plot(x_index_history, Y_predict_history[:, 3],
         color='blue', label='Predict Close Price')
plt.xticks(x_tick_history, date_label_history)
plt.title('Real Close Price vs Predict Close Price [History]')
plt.ylabel('Close Price')
plt.xlabel(f'Date [Simulated Today is {today}]')
plt.legend()
plt.show()

# visualise the future real close price
plt.figure(figsize=(18, 9))
plt.plot(x_index_future, Y_real_future['Close'],
         color='red', label='Real Close Price')
plt.xticks(x_tick_future, date_label_future)
plt.title('Real Close Price vs Predict Close Price [Future]')
plt.ylabel('Close Price')
plt.xlabel(f'Date [Simulated Today is {today}]')
plt.legend()
plt.show()

# visualise the future prediction close price
plt.figure(figsize=(18, 9))
plt.plot(x_index_future, Y_real_future['Close'],
         color='red', label='Real Close Price')
plt.plot(x_index_future, Y_predict_future[:, 3],
         color='blue', label='Predict Close Price')
plt.xticks(x_tick_future, date_label_future)
plt.title('Real Close Price vs Predict Close Price [Future]')
plt.ylabel('Close Price')
plt.xlabel(f'Date [Simulated Today is {today}]')
plt.legend()
plt.show()

BuySellScorer = [9999999999999999, 0, 0, 0]
# [lowest price that come first, highest price at last, day to buy(index), day to sell(index)]

for i in x_index_future:
    # define now looping price
    now_price = Y_predict_future[i, 3]

    # check lowest price that come first
    if now_price < BuySellScorer[0] and i < BuySellScorer[3]:
        BuySellScorer[0] = now_price
        BuySellScorer[2] = i

    # check highest price that come later
    if now_price > BuySellScorer[1] and i > BuySellScorer[2]:
        BuySellScorer[1] = now_price
        BuySellScorer[3] = i

start_money = 30000

date_to_buy = all_date_future[BuySellScorer[2]]
buy_price = Y_predict_future[BuySellScorer[2], 3]
btc_buy = start_money / buy_price
fee_buy = start_money / buy_price * (0.25 / 100)
real_btc = btc_buy - fee_buy

date_to_sell = all_date_future[BuySellScorer[3]]
sell_price = Y_predict_future[BuySellScorer[3], 3]
usd_sell = sell_price * real_btc
fee_sell = sell_price * real_btc * (0.25 / 100)
real_usd = usd_sell - fee_sell

print(f'\nToday is : {today}')

print(f'\nWe have {start_money} $')

print(f'\nWe buy bitcoin at {date_to_buy} with cost {buy_price} $ : 1 BTC')
print(f'We can buy {btc_buy} BTC')
print(f'Fee 0.25% : {fee_buy} BTC')
print(f'We have {real_btc} BTC')
print(f'\nWe sell bitcoin at {date_to_sell} with cost {sell_price} $ : 1 BTC')
print(f'We can sell {usd_sell} $')
print(f'Fee 0.25% : {fee_sell} $')
print(f'We have {real_usd} $')
print(f'\nWe gain {real_usd - start_money} $ for profits')

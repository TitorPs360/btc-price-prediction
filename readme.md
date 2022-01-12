## About BTC Price Prediction

_BTCPricePrediction_ - a project for training an AI (Classification, LSTM) to predict the BTC price, and find when we but or sell.

## Youtube

For more information can be seen in my [video](https://youtu.be/biaRYkLqCec) on YouTube.

[![new_thumb](https://github.com/TitorPs360/btc-price-prediction/blob/main/fig/cover.png)](https://youtu.be/biaRYkLqCec)

## Requirements

- Anaconda with Python 3
- Git

## Install

```
git clone https://github.com/TitorPs360/btc-price-prediction
cd btc-price-prediction
conda create --name <env> --file requirements.txt
```

## Usage

1. Open CMD or Terminal

   ![alt text](https://github.com/TitorPs360/btc-price-prediction/blob/main/fig/step1.png?raw=true)

2. Activate anaconda environment

   ```
   conda activate <env>
   ```

3. Run training script

   - For Training Classification

   ```
   python train_classification.py
   ```

   - For Training LSTM

   ```
   python train_lstm.py
   ```

4. Run test script script

   - For Testing LSTM

   ```
   python test_classification.py
   ```

   - For Testing LSTM

   ```
   python test_lstm.py
   ```

5. Enjoy your result

## Classification Result

- Evaluate Score for Classification
  ![alt text](https://github.com/TitorPs360/btc-price-prediction/blob/main/fig/Classification_result.png?raw=true)

## LSTM Result

- Real History [simulated] vs Predicted History

  ![alt text](https://github.com/TitorPs360/btc-price-prediction/blob/main/fig/history_predict_close_price_btc.png?raw=true)

- Real Future [simulated] vs Predicted Future

  ![alt text](https://github.com/TitorPs360/btc-price-prediction/blob/main/fig/future_predict_close_price_btc.png?raw=true)

- Buy and Sell Result

  ![alt text](https://github.com/TitorPs360/btc-price-prediction/blob/main/fig/LSTM_result.png?raw=true)

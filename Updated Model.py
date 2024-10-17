import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

ticker = input("Enter ticker symbol: ")
s_date = input("Enter start date in YYYY-MM-DD format: ")
e_date = input("Enter end date in YYYY-MM-DD format: ")

data = yf.download(ticker, start=s_date, end=e_date)#data range

ma_100_days = data.Close.rolling(100).mean()#moving Averages

plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='100-Day Moving Average')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
plt.show()

data['Change'] = data['Close'] - data['Open']#Daily Change

sequence_length = 60

scaler = MinMaxScaler(feature_range=(0, 1)

data_scaled = scaler.fit_transform(data)

def create_sequences(data, sequence_length):
  x, y = [], []
  for i in range(len(data) - sequence_length):
    x.append(data_scaled[i:i + sequence_length])
    y.append(data_scaled[i + sequence_length, -1])#last column for closing price
  return np.array(x), np.array(y)

x, y = create_sequences(data_scaled, sequence_length)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create LSTM Model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(sequence_length, data.shape[1])))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')


model.fit(x_train, y_train, epochs=5, batch_size=16, verbose=1)#train model

y_predicted = model.predict(x_test)#predicting data

y_predicted = y_predicted * scaler.scale_[data.shape[1] - 1] + scaler.min_[data.shape[1] - 1]
y_test = y_test * scaler.scale_[data.shape[1] - 1] + scaler.min_[data.shape[1] - 1]

mse = np.mean((y_predicted - y_test)**2)
print("Mean Squared Error:", mse)#calculating mse

model.save('Stock_Prediction_Model.h5')#save model

plt.figure(figsize=(10, 8))
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.plot(y_test, 'b', label='Actual Price')
plt.legend()
plt.show()

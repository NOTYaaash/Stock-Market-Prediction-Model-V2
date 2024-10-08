import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mysql.connector as sql

data = yf.download('AAPL', start='2000-01-01', end='2024-12-31') # data range

ma_100_days = data.Close.rolling(100).mean()

plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r')
plt.plot(data.Close, 'g')
plt.show()

ma_200_days = data.Close.rolling(200).mean()

plt.figure(figsize=(8, 6))
plt.plot(ma_200_days, 'r')
plt.plot(data.Close, 'g')
plt.show()

data['Change'] = data['Close'] - data['Open']#data parameters

scaler = MinMaxScaler(feature_range=(0, 1))#scaling
data_scaled = scaler.fit_transform(data)

sequence_length = 60 # sequence length

def create_sequences(data, sequence_length): #sequences for training and testing
  x, y = [], []
  for i in range(len(data) - sequence_length):
    x.append(data[i:i + sequence_length])
    y.append(data[i + sequence_length, 0])  #next day's closing price
  return np.array(x), np.array(y)

x_train, x_test, y_train, y_test = train_test_split(data_scaled[:-sequence_length], data_scaled[sequence_length:], test_size=0.2)

x_train, x_test = np.array(x_train), np.array(x_test)
y_train, y_test = np.array(y_train), np.array(y_test)

model = Sequential()#LSTM model
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(sequence_length, data.shape[1])))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)#training model

y_predicted = model.predict(x_test)#evaluateing model on test data
y_predicted = y_predicted * scaler.scale_[0] + scaler.min_[0]  #inverse transform
y_test = y_test * scaler.scale_[0] + scaler.min_[0]
mse = np.mean((y_predicted - y_test)**2)
print("Mean Squared Error:", mse)

#prediction on new data

#save model
model.save('Stock_Prediction_Model.h5')

plt.figure(figsize=(10, 8))

y_predicted = model.predict(x_test)  # Evaluate on test data
y_predicted = y_predicted * scaler.scale_[0] + scaler.min_[0]  # Inverse transform


y = data['Close'].to_numpy()  

plt.figure(figsize=(10, 8))

# Reshape y_predict if necessary based on its actual shape
y_predict_reshaped = np.tile(y_predicted, (1, data.shape[1], 1)) 

plt.plot(y_predict_reshaped[0, :, 0], 'r', label='Predicted Price')  
plt.plot(y, 'g', label='Original Price')

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.ylim(min(np.min(y), np.min(y_predicted)), max(np.max(y), np.max(y_predicted)))
plt.show()

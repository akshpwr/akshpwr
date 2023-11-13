```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the stock symbol and fetch historical data
stock_symbol = "AAPL"
data = yf.download(stock_symbol, start="2022-01-01", end="2023-01-01")

# Use only the closing prices for simplicity
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Function to create input sequences for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

# Define hyperparameters
sequence_length = 10
epochs = 50
batch_size = 32

# Create sequences for training
X_train, y_train = create_sequences(prices_scaled, sequence_length)

# Reshape data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Make predictions
test_data = prices_scaled[-sequence_length:]
test_data = test_data.reshape((1, sequence_length, 1))
predicted_price = model.predict(test_data)
predicted_price = scaler.inverse_transform(predicted_price)

# Print the predicted price
print(f"Predicted Price for {stock_symbol}: ${predicted_price[0][0]:.2f}")
```

Keep in mind that this is a basic example, and you may need to adjust parameters and preprocess the data more thoroughly for real-world applications.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error
### fetch bitcoin data
crypto_currency = 'BTC-USD'

start = dt.datetime.now() - dt.timedelta(days=730)
end = dt.datetime.now()

try:
    # Retrieve cryptocurrency data with hourly interval
    data = yf.download(crypto_currency, start=start, end=end, interval='60m')
    
    print(data)
except Exception as e:
    print(f"Error retrieving data: {e}")
### prepare data
# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the data
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Define prediction hours (e.g., using the past 60 hours to predict the next hour)
prediction_hours = 60

# Prepare the training data
x_train, y_train = [], []

for x in range(prediction_hours, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_hours:x, 0])
    y_train.append(scaled_data[x, 0])

# Convert the lists to NumPy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the input data to the required shape for the model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Print the shapes to verify
print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')
### create neural network
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

### testing the data
# Define the cryptocurrency and against currency
crypto_currency = 'BTC-USD'

# Define the date range for testing
start = dt.datetime.now() - dt.timedelta(days=60)
test_end = dt.datetime.now()

# Retrieve test data with hourly intervals
test_data = yf.download(crypto_currency, start=start, end=test_end, interval='60m')

# Get the actual prices
actual_prices = test_data['Close'].values

# Assuming `data` contains your training data
# Combine the training data with the test data
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

# Prepare model inputs
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_hours:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Prepare the test data for predictions
x_test = []

for x in range(prediction_hours, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_hours:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Print the predicted prices
print(predicted_prices)
plt.plot(actual_prices, color='black', label=f'Actual {crypto_currency} price')
plt.plot(predicted_prices, color='green', label=f'Predicted {crypto_currency} price')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel(f'{crypto_currency} Price')
plt.legend(loc='upper left')
plt.show()
### predict next day
# Number of hours to predict (10 days * 24 hours)
num_hours_to_predict = 24

# Initialize the input data with the 240 hours prior to the last 240 hours of the actual data
start_index = len(model_inputs) - prediction_hours - num_hours_to_predict
real_data = model_inputs[start_index:start_index + prediction_hours, 0]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (1, real_data.shape[0], 1))

predicted_prices = []

for _ in range(num_hours_to_predict):
    # Predict the next hour's price
    predicted = model.predict(real_data)
    predicted_price = scaler.inverse_transform(predicted)[0][0]
    predicted_prices.append(predicted_price)
    
    # Update the input data with the new prediction
    new_input = np.append(real_data[0][1:], [[predicted[0][0]]], axis=0)
    real_data = np.reshape(new_input, (1, new_input.shape[0], 1))

# Extract the actual prices for the last 240 hours for comparison
actual_prices = scaler.inverse_transform(model_inputs[-num_hours_to_predict:].reshape(-1, 1)).flatten()

# Print the predicted prices for the next 240 hours
print(f'Predicted prices for the next {num_hours_to_predict} hours: {predicted_prices}')

# Print the actual prices for the last 240 hours
print(f'Actual prices for the last {num_hours_to_predict} hours: {actual_prices}')

# Calculate and print the mean squared error
mse = mean_squared_error(actual_prices, predicted_prices)
print(f'Mean Squared Error: {mse}')
# Optionally, plot the predicted and actual prices for comparison
plt.figure(figsize=(14, 5))
plt.plot(range(num_hours_to_predict), predicted_prices, color='red', linestyle='dashed', label='Predicted Prices')
plt.plot(range(num_hours_to_predict), actual_prices, color='blue', linestyle='solid', label='Actual Prices')
plt.title('Bitcoin Price Prediction vs Actual Prices')
plt.xlabel('Hours')
plt.ylabel('Price (USD)')
plt.legend()
plt.xticks(range(0, num_hours_to_predict, 24), rotation=45)  # Label every 24 hours
plt.show()
import os
import pandas as pd
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('always')  # Temporarily show warnings for debugging

# Load and preprocess data
try:
    data = pd.read_csv("../data/SIH Data.csv")
except FileNotFoundError:
    print("Error: The file '../data/SIH Data.csv' was not found.")
    raise

# Ensure 'Day' and 'Time' columns are properly filled
data['Day'] = data['Day'].ffill()
data['Datetime'] = pd.to_datetime(data['Day'] + ' ' + data['Time'], format='%b-%d %H:%M', errors='coerce')
if data['Datetime'].isna().sum() > 0:
    print(f"Warning: There are {data['Datetime'].isna().sum()} NaT values in the Datetime column.")
data.set_index('Datetime', inplace=True)
data.drop(columns=['Day', 'Time'], inplace=True)
data = data.dropna(how='any', axis=0)

# Split data into train and test sets
train_size = 0.8  # 80% of data for training, 20% for testing
split_index = int(len(data) * train_size)

# Splitting the data chronologically
train, test = data[:split_index], data[split_index:]

# Train the SARIMAX model
ts = train['Load (MW)']
exog_train = train[['Temp', 'Rain(mm)', 'Gust(km/hr)', 'Rain%', 'IsHoliday']]
exog_test = test[['Temp', 'Rain(mm)', 'Gust(km/hr)', 'Rain%', 'IsHoliday']]

model = SARIMAX(ts, exog=exog_train, order=(5, 1, 0))
model_fit = model.fit()

# Save the trained model
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, 'sarimax_model.pkl')
with open(model_path, 'wb') as file:
    pickle.dump(model_fit, file)

print(f"Model saved to {model_path}")

# Forecasting and displaying results
forecast = model_fit.get_forecast(steps=len(test), exog=exog_test)
forecast_series = pd.Series(forecast.predicted_mean.values, index=test.index)

for z, i in zip(test.index[-4:], forecast.predicted_mean.values[-4:]):
    print(f"{z} \t {i:.2f}")

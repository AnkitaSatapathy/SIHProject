# %%
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('data/SIH Data.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
data.set_index('Datetime', inplace=True)
data = data.dropna(how='any',axis=0)

train_size = 0.8  # 80% of data for training, 20% for testing
split_index = int(len(data) * train_size)

# Splitting the data chronologically
train, test = data[:split_index], data[split_index:]

from statsmodels.tsa.statespace.sarimax import SARIMAX

ts = train['Load (MW)']
exog_train = train[['Temp', 'Rain(mm)', 'Gust(km/hr)', 'Rain%', 'IsHoliday']]
exog_test = test[['Temp', 'Rain(mm)', 'Gust(km/hr)', 'Rain%', 'IsHoliday']]

model = SARIMAX(ts, exog=exog_train, order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.get_forecast(steps=len(test), exog=exog_test)
forecast_series = pd.Series(forecast.predicted_mean.values, index=test.index)

def getNewValues():
    return forecast.predicted_mean.values[-1]
    #for z,i in zip(test.index[-4:], forecast.predicted_mean.values[-4:]):
    #    print(f"{i:.2f}")
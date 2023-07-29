import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('EWS M.xlsx')
print(data.head(5))
data.plot(figsize=(12,6))
# plt.plot(data['discharge'],data['Month'])
# plt.show()
# data['Month'] = pd.to_datetime(data['Month'])

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data['discharge'], model='ad',period=12)
result.plot()
# plt.show()

from statsmodels.tsa.stattools import adfuller
result = adfuller(data.discharge.dropna())
print("adf stastics: ",result[0])
print("p-value: ",result[1])
print("Critical Values: ",result[4])
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]
from statsmodels.tsa.arima.model import ARIMA
arima_model = ARIMA(train_data['discharge'],order=(1,1,3))
model = arima_model.fit()
print(model.summary())
mean_actual = data['discharge'].mean()
# Predict future values
forecasted_values = model.predict(start=test_data.index[0], end=test_data.index[-1])
tss = ((data['discharge'] - mean_actual) ** 2).sum()
rss = ((data['discharge'] - forecasted_values) ** 2).sum()
r_squared = 1 - (rss / tss)
print("r2 value of arima: ",r_squared)
#mae,nse,rmse
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error,r2_score
print("MAE: ",mean_absolute_error(test_data['discharge'],forecasted_values))
print("MSE: ",mean_squared_error(test_data['discharge'],forecasted_values))
print("RMSE: ",np.sqrt(mean_squared_error(test_data['discharge'],forecasted_values)))
#nse
predicted_values =  model.predict(start=data.index[0], end=data.index[-1])
dc = pd.DataFrame(forecasted_values)
dc.to_excel('arima_predicted.xlsx')
test_data.to_excel('actual_sss.xlsx')



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
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error,r2_score
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

# sarima implementation
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(train_data['discharge'],order=(1,1,3),seasonal_order=(1,1,2,12))
model_fit = model.fit()
print(model_fit.summary())
mean_actual = data['discharge'].mean()
# Predict future values
forecasted_values1 = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])
tss1 = ((data['discharge'] - mean_actual) ** 2).sum()
rss1 = ((data['discharge'] - forecasted_values1) ** 2).sum()
r_squared1 = 1 - (rss1 / tss1)
print("r2 value Sarima: ",r_squared1)
print("MAE: ",mean_absolute_error(test_data['discharge'],forecasted_values1))
print("MSE: ",mean_squared_error(test_data['discharge'],forecasted_values1))
print("RMSE: ",np.sqrt(mean_squared_error(test_data['discharge'],forecasted_values1)))
dc = pd.DataFrame(forecasted_values1)
dc.to_excel('sarima_pred.xlsx')
test_data.to_excel('actual_data.xlsx')
predictions = model_fit.predict()
predictions.plot()
# plt.show()


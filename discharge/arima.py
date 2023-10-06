import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('DisCharge.xlsx')
print(data.head(5))
data.plot(figsize=(12,6))
# plt.plot(data['discharge'],data['Month'])
# plt.show()
data['Month'] = pd.to_datetime(data['Month'])

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data['discharge'], model='ad',period=12)
result.plot()
plt.show()

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
predictions = model_fit.predict()
predictions.plot()
plt.show()

#LSTM

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.optim as optim

import torch.utils.data as data



df = pd.read_excel('month_avg.xlsx')

timeseries = df[["discharge"]].values.astype('float32')



# train-test split for time series

train_size = int(len(timeseries) * 0.67)

test_size = len(timeseries) - train_size

train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):

    """Transform a time series into a prediction dataset

    

    Args:

        dataset: A numpy array of time series, first dimension is the time steps

        lookback: Size of window for prediction

    """

    X, y = [], []

    for i in range(len(dataset)-lookback):

        feature = dataset[i:i+lookback]

        target = dataset[i+1:i+lookback+1]

        X.append(feature)

        y.append(target)

    return torch.tensor(X), torch.tensor(y)

 

lookback = 4

X_train, y_train = create_dataset(train, lookback=lookback)

X_test, y_test = create_dataset(test, lookback=lookback)

 

class AirModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)

        self.linear = nn.Linear(50, 1)

    def forward(self, x):

        x, _ = self.lstm(x)

        x = self.linear(x)

        return x

 

model = AirModel()

optimizer = optim.Adam(model.parameters())

loss_fn = nn.MSELoss()

loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

 

n_epochs = 2000

for epoch in range(n_epochs):

    model.train()

    for X_batch, y_batch in loader:

        y_pred = model(X_batch)

        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    # Validation

    if epoch % 100 != 0:

        continue

    model.eval()

    with torch.no_grad():

        y_pred = model(X_train)

        train_rmse = np.sqrt(loss_fn(y_pred, y_train))

        y_pred = model(X_test)

        test_rmse = np.sqrt(loss_fn(y_pred, y_test))

        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

 

with torch.no_grad():

    # shift train predictions for plotting

    train_plot = np.ones_like(timeseries) * np.nan

    y_pred = model(X_train)

    y_pred = y_pred[:, -1, :]

    train_plot[lookback:train_size] = model(X_train)[:, -1, :]

    # shift test predictions for plotting

    test_plot = np.ones_like(timeseries) * np.nan

    test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]

# plot

plt.plot(timeseries)

plt.plot(train_plot, c='r')

plt.plot(test_plot, c='g')

plt.show()

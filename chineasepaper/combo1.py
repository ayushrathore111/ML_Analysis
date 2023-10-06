import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the dataset (replace 'your_dataset.csv' with your actual dataset)
dataset = pd.read_excel('load.xlsx')

# Convert datetime columns to appropriate date format (if applicable)
# For example, if the datetime column is named 'date_column':
# dataset['date_column'] = pd.to_datetime(dataset['date_column'])

X= dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
# Forecasting horizon (number of periods into the future)
n_forecast = 12  # Forecasting 12 months (1 year) into the future

# Remove datetime columns from the features (X) if present
X = X.select_dtypes(include=['float64', 'int64'])

# Train the ANN model
ann_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
ann_model.fit(X, y)

# Forecast using ANN model
ann_forecast = ann_model.predict(X)  # Forecast for the training data

# Train the SARIMA model
sarima_order = (1, 0, 1)
sarima_seasonal_order = (1, 0, 2, 12)
sarima_model = SARIMAX(y, order=sarima_order, seasonal_order=sarima_seasonal_order)
sarima_fit = sarima_model.fit(disp=False)

# Forecast using SARIMA model
sarima_forecast = sarima_fit.forecast(steps=n_forecast)  # Forecast for the next n_forecast periods

# Combine the forecasts using a weighted average or other suitable method
# For example, you can use equal weights for both models:
hybrid_forecast = (ann_forecast[-n_forecast:] + sarima_forecast) / 2

# Now you have the hybrid_forecast as the final flood forecasting result for the next n_forecast periods.

# Calculate R-squared (R2)
r2_hybrid = r2_score(y[-n_forecast:], hybrid_forecast)
print("R2 Score for Hybrid Model:", r2_hybrid)

# Calculate Mean Squared Error (MSE)
mse_hybrid = mean_squared_error(y[-n_forecast:], hybrid_forecast)
print("MSE for Hybrid Model:", mse_hybrid)

# Calculate Mean Absolute Error (MAE)
mae_hybrid = mean_absolute_error(y[-n_forecast:], hybrid_forecast)
print("MAE for Hybrid Model:", mae_hybrid)

# Calculate Root Mean Squared Error (RMSE)
rmse_hybrid = mean_squared_error(y[-n_forecast:], hybrid_forecast, squared=False)
print("RMSE for Hybrid Model:", rmse_hybrid)

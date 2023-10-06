
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Read data from Excel file
data = pd.read_excel('load.xlsx')

# Separate input features (X) and output values (y)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

# Define a function to create the ANN model
def create_model(units=50, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(units, input_dim=X_train.shape[1], activation=activation))
    model.add(Dense(1))  # Output layer for regression task
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Create the KerasRegressor wrapper
ann_regressor = KerasRegressor(build_fn=create_model, verbose=0)

# Define the parameter grid for tuning
param_grid = {
    'units': [32, 64, 128],
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [16, 32],
    'epochs': [50, 100]
}

# Perform Grid Search cross-validation for hyperparameter tuning
grid_search = GridSearchCV(ann_regressor, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict on the test data
y_pred = grid_search.best_estimator_.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# Calculate Coefficient of Determination (R-squared)
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

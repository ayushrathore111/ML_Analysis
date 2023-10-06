import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
import xgboost as xgb

# Load data from Excel file
file_path = 'load.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Assuming your features are in columns 'X1', 'X2', ..., 'Xn' and target is in column 'target'
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# Cross-validation for AdaBoostRegressor
ada_model = AdaBoostRegressor(n_estimators=50)  # You can tune n_estimators
ada_scores = -cross_val_score(ada_model, X, y, cv=5, scoring='neg_mean_squared_error')  # Negative RMSE
ada_rmse_values = np.sqrt(ada_scores)
print("AdaBoost Cross-Validation RMSE:", np.sqrt(np.mean(ada_rmse_values)))

# Cross-validation for BaggingRegressor
bagging_model = BaggingRegressor(n_estimators=50)  # You can tune n_estimators
bagging_scores = -cross_val_score(bagging_model, X, y, cv=5, scoring='neg_mean_squared_error')  # Negative RMSE
bagging_rmse_values = np.sqrt(bagging_scores)
print("Bagging Cross-Validation RMSE:", np.sqrt(np.mean(bagging_rmse_values)))

# Cross-validation for XGBRegressor
xgb_model = xgb.XGBRegressor(n_estimators=50)  # You can tune n_estimators and other hyperparameters
xgb_scores = -cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')  # Negative RMSE
xgb_rmse_values = np.sqrt(xgb_scores)
print("XGBoost Cross-Validation RMSE:", np.sqrt(np.mean(xgb_rmse_values)))

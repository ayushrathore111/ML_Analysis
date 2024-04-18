# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

# Assuming you have your dataset loaded into a pandas DataFrame called 'data'
data= pd.read_excel('topic39.xlsx')
# Splitting the dataset into features (X) and target variable (y)
X = data.iloc[:,:-1] # Features
y = data.iloc[:,-1]  # Target variable


import seaborn as sns
import matplotlib.pyplot as plt
# Compute correlation matrix
corr_matrix = data.corr()

# Visualize correlation matrix using heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

plt.show()

# Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardizing the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Ridge Regression
# ridge = Ridge(alpha=1.0)  # You can adjust the regularization strength by changing the alpha parameter
# ridge.fit(X_train_scaled, y_train)

# # Predictions
# ridge_train_predictions = ridge.predict(X_train_scaled)
# ridge_test_predictions = ridge.predict(X_test_scaled)

# # Lasso Regression
# lasso = Lasso(alpha=1.0)  # You can adjust the regularization strength by changing the alpha parameter
# lasso.fit(X_train_scaled, y_train)

# # Predictions
# lasso_train_predictions = lasso.predict(X_train_scaled)
# lasso_test_predictions = lasso.predict(X_test_scaled)

# # Evaluation
# ridge_train_rmse = np.sqrt(mean_squared_error(y_train, ridge_train_predictions))
# ridge_test_rmse = np.sqrt(mean_squared_error(y_test, ridge_test_predictions))
# print("Ridge Regression Train RMSE:", ridge_train_rmse)
# print("Ridge Regression Test RMSE:", ridge_test_rmse)

# lasso_train_rmse = np.sqrt(mean_squared_error(y_train, lasso_train_predictions))
# lasso_test_rmse = np.sqrt(mean_squared_error(y_test, lasso_test_predictions))
# print("Lasso Regression Train RMSE:", lasso_train_rmse)
# print("Lasso Regression Test RMSE:", lasso_test_rmse)

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# Read data from Excel file
data = pd.read_excel('ecc.xlsx')

# Separate input features (X) and output values (y)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Create the Bagging model with a base estimator (e.g., Decision Tree)
base_estimator = DecisionTreeRegressor(max_depth=3)
bagging_model = BaggingRegressor(base_estimator=base_estimator)

# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': [0.5, 0.7, 0.9],
    'max_features': [0.5, 0.7, 0.9],
}

# Grid search with cross-validation
grid_search = GridSearchCV(bagging_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)

# Best hyperparameters and their corresponding score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_
print("Best Hyperparameters for Bagging:", best_params)
print("Best Score:", best_score)

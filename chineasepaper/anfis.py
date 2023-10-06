import numpy as np
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
data = pd.read_excel('load.xlsx')

# Separate the data into input X and output Y
# data=data.dropna()
X = data.iloc[:, :-1].values.reshape(-1,1)
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build ANFIS model
from sklearn.base import BaseEstimator, RegressorMixin

class ANFIS(BaseEstimator, RegressorMixin):
    def __init__(self, num_mfs=5, epochs=50, learning_rate=0.01):
        self.num_mfs = num_mfs
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.fuzzy_models = []
        for i in range(self.num_mfs):
            m = fuzz.trimf(X[:, 0], [np.percentile(X, i*100/self.num_mfs), np.percentile(X, (i+1)*100/self.num_mfs), np.percentile(X, (i+2)*100/self.num_mfs)])
            self.fuzzy_models.append(m)

        X_fuzzy = np.column_stack([fuzz.interp_membership(X[:, 0], m, X[:, 0]) for m in self.fuzzy_models])

        self.weights = np.linalg.inv(X_fuzzy.T @ X_fuzzy) @ X_fuzzy.T @ y
        self.predicted = X_fuzzy @ self.weights

    def predict(self, X):
        X_fuzzy = np.column_stack([fuzz.interp_membership(X[:, 0], m, X[:, 0]) for m in self.fuzzy_models])
        return X_fuzzy @ self.weights

anfis_model = ANFIS()
anfis_model.fit(X_train, y_train)

# Make predictions
y_pred = anfis_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

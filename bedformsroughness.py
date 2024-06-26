
"""bedformsroughness.ipynb"""

import numpy as np 
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("mydata.csv")
data

#put the dataset into dataframe
df = pd.DataFrame(data)

# Split the data into features (X) and target (y)
X = df.drop('n(output parameter manning coefficient)', axis=1)
y = df['n(output parameter manning coefficient)']



from sklearn.linear_model import LinearRegression

# Create and train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Make predictions
y_pred = linear_model.predict(X)

# Calculate evaluation metrics
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rrse = np.sqrt(mse) / (y.max() - y.min())
rae = mae / (y.max() - y.min())

# Print the evaluation metrics
print("Linear Regression - R2 Score:", r2)
print("Linear Regression - RMSE:", rmse)
print("Linear Regression - MSE:", mse)
print("Linear Regression - MAE:", mae)
print("Linear Regression - RRSE:", rrse)
print("Linear Regression - RAE:", rae)

from sklearn.tree import DecisionTreeRegressor

# Create and train the regression tree model
tree_model = DecisionTreeRegressor()
tree_model.fit(X, y)

# Make predictions
y_pred = tree_model.predict(X)

# Calculate evaluation metrics
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rrse = np.sqrt(mse) / (y.max() - y.min())
rae = mae / (y.max() - y.min())

# Print the evaluation metrics
print("Regression Tree - R2 Score:", r2)
print("Regression Tree - RMSE:", rmse)
print("Regression Tree - MSE:", mse)
print("Regression Tree - MAE:", mae)
print("Regression Tree - RRSE:", rrse)
print("Regression Tree - RAE:", rae)

from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = mean_squared_error(y_test, y_pred_lasso, squared=False)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
rrse_lasso = np.sqrt(mse_lasso) / (y_test.max() - y_test.min())
rae_lasso = mae_lasso / (y_test.max() - y_test.min())

print("Lasso Regression - R2 Score:", r2_lasso)
print("Lasso Regression - RMSE:", rmse_lasso)
print("Lasso Regression - MSE:", mse_lasso)
print("Lasso Regression - MAE:", mae_lasso)
print("Lasso Regression - RRSE:", rrse_lasso)
print("Lasso Regression - RAE:", rae_lasso)

from sklearn.ensemble import AdaBoostRegressor

adaboost_model = AdaBoostRegressor()
adaboost_model.fit(X_train, y_train)
y_pred_adaboost = adaboost_model.predict(X_test)

r2_adaboost = r2_score(y_test, y_pred_adaboost)
rmse_adaboost = mean_squared_error(y_test, y_pred_adaboost, squared=False)
mse_adaboost = mean_squared_error(y_test, y_pred_adaboost)
mae_adaboost = mean_absolute_error(y_test, y_pred_adaboost)
rrse_adaboost = np.sqrt(mse_adaboost) / (y_test.max() - y_test.min())
rae_adaboost = mae_adaboost / (y_test.max() - y_test.min())

print("Adaboost Regression - R2 Score:", r2_adaboost)
print("Adaboost Regression - RMSE:", rmse_adaboost)
print("Adaboost Regression - MSE:", mse_adaboost)
print("Adaboost Regression - MAE:", mae_adaboost)
print("Adaboost Regression - RRSE:", rrse_adaboost)
print("Adaboost Regression - RAE:", rae_adaboost)

from sklearn.ensemble import BaggingRegressor

bagging_model = BaggingRegressor()
bagging_model.fit(X_train, y_train)
y_pred_bagging = bagging_model.predict(X_test)

r2_bagging = r2_score(y_test, y_pred_bagging)
rmse_bagging = mean_squared_error(y_test, y_pred_bagging, squared=False)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
mae_bagging = mean_absolute_error(y_test, y_pred_bagging)
rrse_bagging = np.sqrt(mse_bagging) / (y_test.max() - y_test.min())
rae_bagging = mae_bagging / (y_test.max() - y_test.min())

print("Bagging Regression - R2 Score:", r2_bagging)
print("Bagging Regression - RMSE:", rmse_bagging)
print("Bagging Regression - MSE:", mse_bagging)
print("Bagging Regression - MAE:", mae_bagging)
print("Bagging Regression - RRSE:", rrse_bagging)
print("Bagging Regression - RAE:", rae_bagging)

from sklearn.ensemble import ExtraTreesRegressor

extra_trees_model = ExtraTreesRegressor()
extra_trees_model.fit(X_train, y_train)
y_pred_extra_trees = extra_trees_model.predict(X_test)

r2_extra_trees = r2_score(y_test, y_pred_extra_trees)
rmse_extra_trees = mean_squared_error(y_test, y_pred_extra_trees, squared=False)
mse_extra_trees = mean_squared_error(y_test, y_pred_extra_trees)
mae_extra_trees = mean_absolute_error(y_test, y_pred_extra_trees)
rrse_extra_trees = np.sqrt(mse_extra_trees) / (y_test.max() - y_test.min())
rae_extra_trees = mae_extra_trees / (y_test.max() - y_test.min())

print("Extra Trees Regression - R2 Score:", r2_extra_trees)
print("Extra Trees Regression - RMSE:", rmse_extra_trees)
print("Extra Trees Regression - MSE:", mse_extra_trees)
print("Extra Trees Regression - MAE:", mae_extra_trees)
print("Extra Trees Regression - RRSE:", rrse_extra_trees)
print("Extra Trees Regression - RAE:", rae_extra_trees)

from sklearn.ensemble import GradientBoostingRegressor

gradient_boosting_model = GradientBoostingRegressor()
gradient_boosting_model.fit(X_train, y_train)
y_pred_gradient_boosting = gradient_boosting_model.predict(X_test)

r2_gradient_boosting = r2_score(y_test, y_pred_gradient_boosting)
rmse_gradient_boosting = mean_squared_error(y_test, y_pred_gradient_boosting, squared=False)
mse_gradient_boosting = mean_squared_error(y_test, y_pred_gradient_boosting)
mae_gradient_boosting = mean_absolute_error(y_test, y_pred_gradient_boosting)
rrse_gradient_boosting = np.sqrt(mse_gradient_boosting) / (y_test.max() - y_test.min())
rae_gradient_boosting = mae_gradient_boosting / (y_test.max() - y_test.min())

print("Gradient Boosting Regression - R2 Score:", r2_gradient_boosting)
print("Gradient Boosting Regression - RMSE:", rmse_gradient_boosting)
print("Gradient Boosting Regression - MSE:", mse_gradient_boosting)
print("Gradient Boosting Regression - MAE:", mae_gradient_boosting)
print("Gradient Boosting Regression - RRSE:", rrse_gradient_boosting)
print("Gradient Boosting Regression - RAE:", rae_gradient_boosting)

from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

r2_knn = r2_score(y_test, y_pred_knn)
rmse_knn = mean_squared_error(y_test, y_pred_knn, squared=False)
mse_knn = mean_squared_error(y_test, y_pred_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
rrse_knn = np.sqrt(mse_knn) / (y_test.max() - y_test.min())
rae_knn = mae_knn / (y_test.max() - y_test.min())

print("K-Nearest Neighbors - R2 Score:", r2_knn)
print("K-Nearest Neighbors - RMSE:", rmse_knn)
print("K-Nearest Neighbors - MSE:", mse_knn)
print("K-Nearest Neighbors - MAE:", mae_knn)
print("K-Nearest Neighbors - RRSE:", rrse_knn)
print("K-Nearest Neighbors - RAE:", rae_knn)

from sklearn.svm import SVR

# Create and train the SVR model
model = SVR()
model.fit(X, y)

# Make predictions using the trained model
y_pred = model.predict(X)

# Evaluate the model
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rrse = np.sqrt(mse) / (y.max() - y.min())
rae = mae / (y.max() - y.min())

print("SVR - R2 Score:", r2)
print("SVR - RMSE:", rmse)
print("SVR - MSE:", mse)
print("SVR - MAE:", mae)
print("SVR - RRSE:", rrse)
print("SVR - RAE:", rae)

from sklearn.ensemble import RandomForestRegressor

# Create and train the Random Forest regression model
model = RandomForestRegressor()
model.fit(X, y)

# Make predictions using the trained model
y_pred = model.predict(X)

# Evaluate the model
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rrse = np.sqrt(mse) / (y.max() - y.min())
rae = mae / (y.max() - y.min())

print("Random Forest Regression - R2 Score:", r2)
print("Random Forest Regression - RMSE:", rmse)
print("Random Forest Regression - MSE:", mse)
print("Random Forest Regression - MAE:", mae)
print("Random Forest Regression - RRSE:", rrse)
print("Random Forest Regression - RAE:", rae)

from sklearn.ensemble import GradientBoostingRegressor

# Create and train the Gradient Boosting regression model
model = GradientBoostingRegressor()
model.fit(X, y)

# Make predictions using the trained model
y_pred = model.predict(X)

# Evaluate the model
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rrse = np.sqrt(mse) / (y.max() - y.min())
rae = mae / (y.max() - y.min())

print("Gradient Boosting Regression - R2 Score:", r2)
print("Gradient Boosting Regression - RMSE:", rmse)
print("Gradient Boosting Regression - MSE:", mse)
print("Gradient Boosting Regression - MAE:", mae)
print("Gradient Boosting Regression - RRSE:", rrse)
print("Gradient Boosting Regression - RAE:", rae)

from xgboost import XGBRegressor

# Create and train the XGBoost regression model
model = XGBRegressor()
model.fit(X, y)

# Make predictions using the trained model
y_pred = model.predict(X)

# Evaluate the model
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rrse = np.sqrt(mse) / (y.max() - y.min())
rae = mae / (y.max() - y.min())

print("XGBoost Regression - R2 Score:", r2)
print("XGBoost Regression - RMSE:", rmse)
print("XGBoost Regression - MSE:", mse)
print("XGBoost Regression - MAE:", mae)
print("XGBoost Regression - RRSE:", rrse)
print("XGBoost Regression - RAE:", rae)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load the dataset
data = pd.read_csv('mydata.csv')

# Split the data into features (X) and target (y)
X = data.drop('n(output parameter manning coefficient)', axis=1)
y = data['n(output parameter manning coefficient)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and configure the M5P model
m5p = DecisionTreeRegressor()

# Train the M5P model
m5p.fit(X_train, y_train)

# Make predictions on the test set
predictions = m5p.predict(X_test)

# Evaluate the model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rrse = np.sqrt(mse) / (y.max() - y.min())
print("M5P Regression - R2 Score:", r2)
print("M5P Regression - RMSE:", rmse)
print("M5P Regression - MSE:", mse)
print("M5P Regression - MAE:", mae)
print("M5P Regression - RRSE:", rrse)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the dataset
data = pd.read_csv('mydata.csv')

# Split the data into features (X) and target (y)
X = data.drop('n(output parameter manning coefficient)', axis=1)
y = data['n(output parameter manning coefficient)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and configure the k*-NN model
k_star = KNeighborsRegressor()

# Train the k*-NN model
k_star.fit(X_train, y_train)

# Make predictions on the test set
predictions = k_star.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rrse = 
print("k*-NN Regression - R2 Score:", r2)
print("k*-NN Regression - RMSE:", rmse)
print("k*-NN Regression - MSE:", mse)
print("k*-NN Regression - MAE:", mae)
print("k*-NN Regression - RRSE:", rrse)
print("k*-NN Regression - RAE:", rae)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the dataset
data = pd.read_csv('mydata.csv')

# Split the data into features (X) and target (y)
X = data.drop('n(output parameter manning coefficient)', axis=1)
y = data['n(output parameter manning coefficient)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and configure the Gaussian Process model
kernel = RBF()
gaussian_process = GaussianProcessRegressor(kernel=kernel)

# Train the Gaussian Process model
gaussian_process.fit(X_train, y_train)

# Make predictions on the test set
predictions = gaussian_process.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("Gaussian Process Regression - R2 Score:", r2)
print("Gaussian Process Regression - RMSE:", rmse)
print("Gaussian Process Regression - MSE:", mse)
print("Gaussian Process Regression - MAE:", mae)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset into a DataFrame
df = pd.read_csv('mydata.csv')

# Split the data into features (X) and target (y)
X = df.drop('n(output parameter manning coefficient)', axis=1)
y = df['n(output parameter manning coefficient)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Naive Bayes classifier
naive_bayes = GaussianNB()

# Train the classifier
naive_bayes.fit(X_train, y_train)

# Make predictions on the test set
y_pred = naive_bayes.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report)

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# Load the dataset into a DataFrame
df = pd.read_csv('mydata.csv')

# Split the data into features (X) and target (y)
X = df.drop('n(output parameter manning coefficient)', axis=1)
y = df['n(output parameter manning coefficient)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the MLPRegressor
mlp_regressor = MLPRegressor()
mlp_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp_regressor.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Split the data into features (X) and target (y)
X = df.drop('n(output parameter manning coefficient)', axis=1)
y = df['n(output parameter manning coefficient)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the MLPRegressor
mlp_regressor = MLPRegressor()
mlp_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp_regressor.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
print

print("neural_network - R2 Score:", r2)
print("neural_network - RMSE:", rmse)
print("neural_network - MSE:", mse)
print("neural_network - MAE:", mae)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Split the data into features (X) and target (y)
X = df.drop('n(output parameter manning coefficient)', axis=1)
y = df['n(output parameter manning coefficient)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features
degree = 2  # Set the degree of polynomial
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Create and train the Polynomial Regression model
poly_regression = LinearRegression()
poly_regression.fit(X_train_poly, y_train)

# Make predictions on the test set
y_pred = poly_regression.predict(X_test_poly)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("poly_regression - R2 Score:", r2)
print("poly_regression - RMSE:", rmse)
print("poly_regression - MSE:", mse)
print("poly_regression - MAE:", mae)


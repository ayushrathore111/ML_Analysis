import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

# Read the CSV file
data = pd.read_excel('ecc.xlsx')

# Separate the data into input X and output Y
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create individual models
LR = LinearRegression()
RF = RandomForestRegressor(max_depth=2, random_state=0)
DT = DecisionTreeRegressor(random_state=0)
GBR = GradientBoostingRegressor(random_state=0)
MLP = MLPRegressor(random_state=1, max_iter=50)
knn_model = KNeighborsRegressor()
extra_trees_model = ExtraTreesRegressor()
bagging_model = BaggingRegressor()
adaboost_model = AdaBoostRegressor()
model = XGBRegressor()

# Fit individual models
LR.fit(X_train, Y_train)
RF.fit(X_train, Y_train)
DT.fit(X_train, Y_train)
GBR.fit(X_train, Y_train)
MLP.fit(X_train, Y_train)
knn_model.fit(X_train, Y_train)
extra_trees_model.fit(X_train, Y_train)
bagging_model.fit(X_train, Y_train)
adaboost_model.fit(X_train, Y_train)
model.fit(X_train, Y_train)

# Make predictions using individual models
Y_pred_LR = LR.predict(X_test)
Y_pred_RF = RF.predict(X_test)
Y_pred_DT = DT.predict(X_test)
Y_pred_GBR = GBR.predict(X_test)
Y_pred_MLP = MLP.predict(X_test)
y_pred_knn = knn_model.predict(X_test)
y_pred_extra_trees = extra_trees_model.predict(X_test)
y_pred_bagging = bagging_model.predict(X_test)
y_pred_adaboost = adaboost_model.predict(X_test)
y_pred = model.predict(X_test)

# Create an ensemble prediction by averaging the predictions
ensemble_prediction = (Y_pred_LR + Y_pred_RF + Y_pred_DT + Y_pred_GBR + Y_pred_MLP + y_pred_knn +
                       y_pred_extra_trees + y_pred_bagging + y_pred_adaboost + y_pred) / 10.0

# Evaluate the ensemble model
r2_ensemble = r2_score(Y_test, ensemble_prediction)
rmse_ensemble = np.sqrt(mean_squared_error(Y_test, ensemble_prediction, squared=False))
mse_ensemble = mean_squared_error(Y_test, ensemble_prediction)
mae_ensemble = mean_absolute_error(Y_test, ensemble_prediction)
rrse_ensemble = np.sqrt(mse_ensemble) / (max(Y_test.max(), Y_train.max()) - min(Y_test.min(), Y_train.min()))
rae_ensemble = mae_ensemble / (max(Y_test.max(), Y_train.max()) - min(Y_test.min(), Y_train.min()))
std_ensemble = np.std(Y_test - ensemble_prediction)

print("Ensemble Model - R2 Score:", r2_ensemble)
print("Ensemble Model - RMSE:", rmse_ensemble)
print("Ensemble Model - MSE:", mse_ensemble)
print("Ensemble Model - MAE:", mae_ensemble)
print("Ensemble Model - RRSE:", rrse_ensemble)
print("Ensemble Model - RAE:", rae_ensemble)
print("Ensemble Model - Standard Deviation:", std_ensemble)

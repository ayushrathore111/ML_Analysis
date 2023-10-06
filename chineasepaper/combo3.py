import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble         import RandomForestRegressor
from sklearn.linear_model     import LinearRegression
from sklearn.tree             import DecisionTreeRegressor
from sklearn.svm              import SVR
from sklearn.ensemble         import GradientBoostingRegressor
from sklearn.neural_network   import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
# Read the CSV file
data = pd.read_excel("y2.xlsx")
# print(data.describe())
# Separate the data into input X and output Y
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=0)

from sklearn.ensemble import ExtraTreesRegressor

extra_trees_model = ExtraTreesRegressor()
extra_trees_model.fit(X_train, Y_train)
y_pred_extra_trees = extra_trees_model.predict(X_test)

GBR =GradientBoostingRegressor(random_state=0)
GBR.fit(X_train, Y_train)
Y_pred_GBR = GBR.predict(X_test)

y_pred = (Y_pred_GBR+y_pred_extra_trees)/2

r2_knn = r2_score(Y_test, y_pred)
rmse_knn = mean_squared_error(Y_test, y_pred, squared=False)
mse_knn = mean_squared_error(Y_test, y_pred)
mae_knn = mean_absolute_error(Y_test, y_pred)
rrse_knn = np.sqrt(mse_knn) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
rae_knn = mae_knn / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))

print("hybrid:LR & RF - R2 Score:", r2_knn)
print("hybrid:LR & RF - RMSE:", rmse_knn)
print("hybrid:LR & RF - MSE:", mse_knn)
print("hybrid:LR & RF - MAE:", mae_knn)
print("hybrid:LR & RF - RRSE:", rrse_knn)
print("hybrid:LR & RF - RAE:", rae_knn)
dc = pd.DataFrame([r2_knn,rmse_knn,mse_knn,mae_knn,rae_knn,rrse_knn])
dc.to_excel('ETR_DTR.xlsx')
# print("K-Nearest Neighbors - Standard Dav:", std_knn)
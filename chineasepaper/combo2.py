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
data = pd.read_excel("load.xlsx")
# print(data.describe())
# Separate the data into input X and output Y
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=0)
LR = LinearRegression()
LR.fit(X, Y)
Y_pred_LR = LR.predict(X)
RF = RandomForestRegressor (n_estimators  =36 ,random_state    =47 ) 
RF.fit(X, Y)
Y_pred_RF = RF.predict(X)
do = pd.DataFrame(Y_test)
do.to_excel("test.xlsx")
y_pred = (Y_pred_RF+Y_pred_LR)/2
dc = pd.DataFrame(y_pred)
dc.to_excel("ETR_BR_pred.xlsx")
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
dc.to_excel('RF_LR.xlsx')
# print("K-Nearest Neighbors - Standard Dav:", std_knn)

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
data = pd.read_excel("ecc.xlsx")
# print(data.describe())
# Separate the data into input X and output Y
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
LR = LinearRegression()
LR.fit(X_train, Y_train)
Y_pred_LR = LR.predict(X_test)
RF = RandomForestRegressor (n_estimators  =36 ,random_state =47 ) 
RF.fit(X_train, Y_train)
Y_pred_RF = RF.predict(X_test)
do = pd.DataFrame(Y_test)
do.to_excel("test.xlsx")
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor

# Create and train the XGBoost regression model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Make predictions using the trained model
y_pred_xg = model.predict(X_test)

GBR =GradientBoostingRegressor(random_state=0)
GBR.fit(X_train, Y_train)
Y_pred_GBR = GBR.predict(X_test)
from sklearn.ensemble import BaggingRegressor
bagging_model = BaggingRegressor()
bagging_model.fit(X_train, Y_train)
y_pred_bagging = bagging_model.predict(X_test)
from sklearn.neighbors import KNeighborsRegressor
DT = DecisionTreeRegressor(random_state=0)
DT.fit(X_train, Y_train)
Y_pred_DT = DT.predict(X_test)
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, Y_train)
y_pred_knn = knn_model.predict(X_test)
from sklearn.ensemble import ExtraTreesRegressor

extra_trees_model = ExtraTreesRegressor()
extra_trees_model.fit(X_train, Y_train)
y_pred_extra_trees = extra_trees_model.predict(X_test)
from sklearn.ensemble import AdaBoostRegressor

adaboost_model = AdaBoostRegressor()
adaboost_model.fit(X_train, Y_train)
y_pred_adaboost = adaboost_model.predict(X_test)
y_pred = (y_pred_adaboost+y_pred_bagging)/2
dc = pd.DataFrame(y_pred)
dc.to_excel("ETR_BR_pred.xlsx")
r2_knn = r2_score(Y_test, y_pred)
rmse_knn = mean_squared_error(Y_test, y_pred, squared=False)
mse_knn = mean_squared_error(Y_test, y_pred)
mae_knn = mean_absolute_error(Y_test, y_pred)
rrse_knn = np.sqrt(mse_knn) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
rae_knn = mae_knn / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
mean_observed = np.mean(Y_test)
residue_knn = Y_test-y_pred
std_knn = np.std(residue_knn)
print(" - R2 Score:", r2_knn)
print("- RMSE:", rmse_knn)
print("- MSE:", mse_knn)
print("- MAE:", mae_knn)
print("- RRSE:", rrse_knn)
print("- RAE:", rae_knn)
VAF_knn = 100 * (1 - (np.var(Y_test - y_pred) / np.var(Y_test)))
print("VAF: ",VAF_knn)
nse_knn = 1 - ( np.sum((Y_test - y_pred)**2) / np.sum((Y_test - mean_observed)**2))
print("NSE: ",nse_knn)
r = np.corrcoef(Y_test, y_pred)[0, 1]
alpha = np.std(y_pred) / np.std(Y_test)
beta = np.mean(y_pred) / np.mean(Y_test)
kge_knn = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
print("KGE:",kge_knn)
dc = pd.DataFrame([r2_knn,rmse_knn,mse_knn,mae_knn,rae_knn,rrse_knn,std_knn,VAF_knn,kge_knn,nse_knn])
dc.to_excel('RF_LR.xlsx')
# print("K-Nearest Neighbors - Standard Dav:", std_knn)

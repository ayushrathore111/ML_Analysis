import pandas as pd
import numpy as np
from sklearn.ensemble         import RandomForestRegressor
from sklearn.linear_model     import LinearRegression
from sklearn.tree             import DecisionTreeRegressor
from sklearn.svm              import SVR
from sklearn.ensemble         import GradientBoostingRegressor
from sklearn.neural_network   import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
# Read the CSV file
data = pd.read_excel('nonrepetitive.xlsx')

# Separate the data into input X and output Y
data=data.dropna()
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(  X, Y, test_size=0.20, random_state=0)
# Create the linear regression model
Y_test.to_excel('actula.xlsx')
LR = LinearRegression()
LR.fit(X_train, Y_train)
Y_pred_LR = LR.predict(X_test)
y_lr_f= LR.predict(X)
dc9 = pd.DataFrame(Y_pred_LR)
dc9.to_excel('lr.xlsx')
residue_lr = Y-y_lr_f
std_lr = np.std(residue_lr)
# Random forest 
RF = RandomForestRegressor(max_depth=2, random_state=0)
RF.fit(X_train, Y_train)
Y_pred_RF = RF.predict(X_test)
y_rf_f= RF.predict(X)
dc8 = pd.DataFrame(Y_pred_RF)
dc8.to_excel('rf.xlsx')
residue_rf = Y-y_rf_f
std_rf = np.std(residue_rf)
# Decision Tree
DT = DecisionTreeRegressor(random_state=0)
DT.fit(X_train, Y_train)
Y_pred_DT = DT.predict(X_test)
print(Y_pred_DT)
y_dt_f= DT.predict(X)
dc7 = pd.DataFrame(Y_pred_DT)
dc7.to_excel('dt.xlsx')
residue_dt = Y-y_dt_f
std_dt = np.std(residue_dt)

# SVR
SVR = SVR(kernel="linear", C=100, gamma=0.1, epsilon=0.1)
SVR.fit(X_train, Y_train)
Y_pred_SVR = SVR.predict(X_test)
y_svr_f= SVR.predict(X)
dc6 = pd.DataFrame(Y_pred_SVR)
dc6.to_excel('svr.xlsx')
residue_svr = Y-y_svr_f
std_svr = np.std(residue_svr)
# GBR
GBR =GradientBoostingRegressor(random_state=0)
GBR.fit(X_train, Y_train)
Y_pred_GBR = GBR.predict(X_test)
y_gbr_f= GBR.predict(X)
dc5 = pd.DataFrame(Y_pred_GBR)
dc5.to_excel('gbr.xlsx')
residue_gbr = Y-y_gbr_f
std_gbr = np.std(residue_gbr)
#MLP
MLP = MLPRegressor(random_state=1, max_iter=50)
MLP.fit(X_train, Y_train)
Y_pred_MLP = MLP.predict(X_test)
y_mlp_f= MLP.predict(X)
dc5 = pd.DataFrame(Y_pred_MLP)
dc5.to_excel('mlp.xlsx')
residue = Y-y_mlp_f
std_mlp = np.std(residue)

from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor()
knn_model.fit(X_train, Y_train)
y_pred_knn = knn_model.predict(X_test)
y_knn_f= knn_model.predict(X)
dc4 = pd.DataFrame(y_pred_knn)
dc4.to_excel('knn.xlsx')
residue_knn = Y-y_knn_f
std_knn = np.std(residue_knn)
r2_knn = r2_score(Y_test, y_pred_knn)
rmse_knn = mean_squared_error(Y_test, y_pred_knn, squared=False)
mse_knn = mean_squared_error(Y_test, y_pred_knn)
mae_knn = mean_absolute_error(Y_test, y_pred_knn)
rrse_knn = np.sqrt(mse_knn) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
rae_knn = mae_knn / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))

print("K-Nearest Neighbors - R2 Score:", r2_knn)
print("K-Nearest Neighbors - RMSE:", rmse_knn)
print("K-Nearest Neighbors - MSE:", mse_knn)
print("K-Nearest Neighbors - MAE:", mae_knn)
print("K-Nearest Neighbors - RRSE:", rrse_knn)
print("K-Nearest Neighbors - RAE:", rae_knn)
print("K-Nearest Neighbors - Standard Dav:", std_knn)



from sklearn.ensemble import ExtraTreesRegressor

extra_trees_model = ExtraTreesRegressor()
extra_trees_model.fit(X_train, Y_train)
y_pred_extra_trees = extra_trees_model.predict(X_test)
y_et_f= extra_trees_model.predict(X)
dc3= pd.DataFrame(y_pred_extra_trees)
dc3.to_excel('etr.xlsx')
residue_et = Y-y_et_f
std_et = np.std(residue_et)
r2_extra_trees = r2_score(Y_test, y_pred_extra_trees)
rmse_extra_trees = mean_squared_error(Y_test, y_pred_extra_trees, squared=False)
mse_extra_trees = mean_squared_error(Y_test, y_pred_extra_trees)
mae_extra_trees = mean_absolute_error(Y_test, y_pred_extra_trees)
rrse_extra_trees = np.sqrt(mse_extra_trees) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
rae_extra_trees = mae_extra_trees / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
print("Extra Trees Regression - R2 Score:", r2_extra_trees)
print("Extra Trees Regression - RMSE:", rmse_extra_trees)
print("Extra Trees Regression - MSE:", mse_extra_trees)
print("Extra Trees Regression - MAE:", mae_extra_trees)
print("Extra Trees Regression - RRSE:", rrse_extra_trees)
print("Extra Trees Regression - RAE:", rae_extra_trees)
print("Extra Trees Regression - SD:", std_et)
from sklearn.ensemble import BaggingRegressor

bagging_model = BaggingRegressor()
bagging_model.fit(X_train, Y_train)
y_pred_bagging = bagging_model.predict(X_test)
y_br_f= bagging_model.predict(X)
dc2 = pd.DataFrame(y_pred_bagging)
dc2.to_excel('br.xlsx')
residue_br = Y-y_br_f
std_br = np.std(residue_br)
r2_bagging = r2_score(Y_test, y_pred_bagging)
rmse_bagging = np.sqrt(mean_squared_error(Y_test, y_pred_bagging, squared=False))
mse_bagging = mean_squared_error(Y_test, y_pred_bagging)
mae_bagging = mean_absolute_error(Y_test, y_pred_bagging)
rrse_bagging = np.sqrt(mse_bagging) /(max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
rae_bagging = mae_bagging / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))

print("Bagging Regression - R2 Score:", r2_bagging)
print("Bagging Regression - RMSE:", rmse_bagging)
print("Bagging Regression - MSE:", mse_bagging)
print("Bagging Regression - MAE:", mae_bagging)
print("Bagging Regression - RRSE:", rrse_bagging)
print("Bagging Regression - RAE:", rae_bagging)
print("Bagging Regression - SD:", std_br)
from sklearn.ensemble import AdaBoostRegressor

adaboost_model = AdaBoostRegressor()
rae_bagging = mae_bagging / (Y_test.max() - Y_test.min())
adaboost_model.fit(X_train, Y_train)
y_pred_adaboost = adaboost_model.predict(X_test)
y_ar_f= adaboost_model.predict(X)
dc = pd.DataFrame(y_pred_adaboost)
dc.to_excel('adaboost.xlsx')
residue_ar = Y-y_ar_f
std_ar = np.std(residue_ar)
r2_adaboost = r2_score(Y_test, y_pred_adaboost)
rmse_adaboost =np.sqrt(mean_squared_error(Y_test, y_pred_adaboost, squared=False))
mse_adaboost = mean_squared_error(Y_test, y_pred_adaboost)
mae_adaboost = mean_absolute_error(Y_test, y_pred_adaboost)
rrse_adaboost = np.sqrt(mse_adaboost) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
rae_adaboost = mae_adaboost / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
print("Adaboost Regression - R2 Score:", r2_adaboost)
print("Adaboost Regression - RMSE:", rmse_adaboost)
print("Adaboost Regression - MSE:", mse_adaboost)
print("Adaboost Regression - MAE:", mae_adaboost)
print("Adaboost Regression - RRSE:", rrse_adaboost)
print("Adaboost Regression - RAE:", rae_adaboost)
print("Adaboost Regression - SD:", std_ar)

from xgboost import XGBRegressor

# Create and train the XGBoost regression model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Make predictions using the trained model
y_pred = model.predict(X_test)
y_pred_f= model.predict(X)
dc1 = pd.DataFrame(y_pred)
dc1.to_excel('xgb.xlsx')
# Evaluate the model
r2 = r2_score(Y_test, y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred, squared=False))
mse = mean_squared_error(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
rrse = np.sqrt(mse) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
rae = mae / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))

print("XGBoost Regression - R2 Score:", r2)
print("XGBoost Regression - RMSE:", rmse)
print("XGBoost Regression - MSE:", mse)
print("XGBoost Regression - MAE:", mae)
print("XGBoost Regression - RRSE:", rrse)
print("XGBoost Regression - RAE:", rae)


from sklearn import metrics
import numpy as np
mae_lr=metrics.mean_absolute_error(Y_test, Y_pred_LR)
mse_lr= metrics.mean_squared_error(Y_test, Y_pred_LR)
rmse_lr=np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_LR))
r2_lr=metrics.r2_score(Y_test, Y_pred_LR)
rrse_lr= np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_LR)) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
rae_lr = mae_lr/ (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
print("\nLinearRegression")
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred_LR))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred_LR))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_LR)))
print('correlation fatcor',metrics.r2_score(Y_test, Y_pred_LR))
# rrse = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_LR)) / (Y_test.max() - Y_test.min())

print('rrse: ',np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_LR)) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min())))
print('rae: ',metrics.mean_absolute_error(Y_test, Y_pred_LR) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min())))
print('SD: ',std_lr)
mae_rf=metrics.mean_absolute_error(Y_test, Y_pred_RF)
mse_rf= metrics.mean_squared_error(Y_test, Y_pred_RF)
rmse_rf=np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_RF))
r2_rf=metrics.r2_score(Y_test, Y_pred_RF)
rrse_rf= np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_RF)) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
rae_rf = mae_lr/ (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
print("\nRandomForestRegressor")
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred_RF))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred_RF))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_RF)))
print('correlation fatcor',metrics.r2_score(Y_test, Y_pred_RF))
print('rrse: ',np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_RF)) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min())))
print('rae: ',metrics.mean_absolute_error(Y_test, Y_pred_RF) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min())))
print('SD: ',std_rf)
print("\nDecisionTreeRegressor")
mae_dtr=metrics.mean_absolute_error(Y_test, Y_pred_DT)
mse_dtr= metrics.mean_squared_error(Y_test, Y_pred_DT)
rmse_dtr=np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_DT))
r2_dtr=metrics.r2_score(Y_test, Y_pred_DT)
rrse_dtr= np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_DT)) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
rae_dtr = mae_dtr/ (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred_DT))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred_DT))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_DT)))
print('correlation fatcor',metrics.r2_score(Y_test, Y_pred_DT))
print('rrse: ',np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_DT)) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min())))
print('rae: ',metrics.mean_absolute_error(Y_test, Y_pred_DT) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min())))
print('SD: ',std_dt)
print("\nSVR")
mae_svr=metrics.mean_absolute_error(Y_test, Y_pred_SVR)
mse_svr= metrics.mean_squared_error(Y_test, Y_pred_SVR)
rmse_svr=np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_SVR))
r2_svr=metrics.r2_score(Y_test, Y_pred_SVR)
rrse_svr= np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_SVR)) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
rae_svr = mae_svr/ (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred_SVR))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred_SVR))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_SVR)))
print('correlation fatcor',metrics.r2_score(Y_test, Y_pred_SVR))
print('rrse: ',np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_SVR)) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min())))
print('rae: ',metrics.mean_absolute_error(Y_test, Y_pred_SVR) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min())))
print('SD: ',std_svr)
print("\nGradientBoostingRegressor")
mae_gbr=metrics.mean_absolute_error(Y_test, Y_pred_GBR)
mse_gbr= metrics.mean_squared_error(Y_test, Y_pred_GBR)
rmse_gbr=np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_GBR))
r2_gbr=metrics.r2_score(Y_test, Y_pred_GBR)
rrse_gbr= np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_GBR)) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
rae_gbr = mae_gbr/ (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred_GBR))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred_GBR))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_GBR)))
print('correlation fatcor',metrics.r2_score(Y_test, Y_pred_GBR))
print('rrse: ',np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_GBR)) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min())))
print('rae: ',metrics.mean_absolute_error(Y_test, Y_pred_GBR) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min())))
print('SD: ',std_gbr)
print("\nMLPRegressor")
mae_mlp=metrics.mean_absolute_error(Y_test, Y_pred_MLP)
mse_mlp= metrics.mean_squared_error(Y_test, Y_pred_MLP)
rmse_mlp=np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_MLP))
r2_mlp=metrics.r2_score(Y_test, Y_pred_MLP)
rrse_mlp= np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_MLP)) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
rae_mlp = mae_mlp/ (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min()))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred_MLP))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred_MLP))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_MLP)))
print('correlation fatcor',metrics.r2_score(Y_test, Y_pred_MLP))
print('rrse: ',np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_MLP)) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min())))
print('rae: ',metrics.mean_absolute_error(Y_test, Y_pred_MLP) / (max(Y_test.max(),Y_train.max()) - min(Y_test.min(),Y_train.min())))
print('SD: ',std_mlp)

cd = pd.DataFrame([r2_knn,rmse_knn,mse_knn,mae_knn,rrse_knn,rae_knn,r2_extra_trees,rmse_extra_trees,mse_extra_trees,mae_extra_trees,rrse_extra_trees,rae_extra_trees,r2_bagging,rmse_bagging,mse_bagging,mae_bagging,rrse_bagging,rae_bagging,r2_adaboost,rmse_adaboost,mse_adaboost,mae_adaboost,rrse_adaboost,rae_adaboost,r2,rmse,mse,mae,rrse,rae,r2_lr,rmse_lr,mse_lr,mae_lr,rrse_lr,rae_lr,r2_rf,rmse_rf,mse_rf,mae_rf,rrse_rf,rae_rf,r2_dtr,rmse_dtr,mse_dtr,mae_dtr,rrse_dtr,rae_dtr,r2_svr,rmse_svr,mse_svr,mae_svr,rrse_svr,rae_svr,r2_gbr,rmse_gbr,mse_gbr,mae_gbr,rrse_gbr,rae_gbr,r2_mlp,rmse_mlp,mse_mlp,mae_mlp,rrse_mlp,rae_mlp])
cd.to_excel("knn_error.xlsx")
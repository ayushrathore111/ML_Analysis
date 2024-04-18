
import pandas as pd
import numpy as np
from sklearn.ensemble         import RandomForestRegressor
from sklearn.linear_model     import LinearRegression
from sklearn.tree             import DecisionTreeRegressor
from sklearn.svm              import SVR
from sklearn.ensemble         import GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.neural_network   import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
# Read the CSV file
data = pd.read_excel('topic39.xlsx')
from sklearn.neighbors import KNeighborsRegressor

# Separate the data into input X and output Y
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=None)
lr_model = LinearRegression()
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
knn_model = KNeighborsRegressor(n_neighbors=5)
etr_model = ExtraTreesRegressor(random_state=42)
dtr_model = DecisionTreeRegressor(random_state=42)
adaboost_model = AdaBoostRegressor(random_state=42)
bagging_model = BaggingRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)

y_pred_knn= knn_model.fit(X_train,y_train).predict(X_test)
y_pred_etr= etr_model.fit(X_train,y_train).predict(X_test)
y_pred_dtr= dtr_model.fit(X_train,y_train).predict(X_test)
y_pred_br= bagging_model.fit(X_train,y_train).predict(X_test)
y_pred_ar= adaboost_model.fit(X_train,y_train).predict(X_test)
y_pred_xg= xgb_model.fit(X_train,y_train).predict(X_test)
y_pred_lr= lr_model.fit(X_train,y_train).predict(X_test)
y_pred_rf= rf_model.fit(X_train,y_train).predict(X_test)
y_pred_gbr= gbr_model.fit(X_train,y_train).predict(X_test)

# LR&XGB
# RF&LR
# GBR&LR
# BR&GBR
# KNN&BR
# LR&BR

lr_xg= (y_pred_knn+y_pred_dtr)/2
rf_lr= (y_pred_ar+y_pred_rf)/2
lr_gbr= (y_pred_gbr+y_pred_lr)/2
br_gbr= (y_pred_br+y_pred_gbr)/2
etr_br= (y_pred_etr+y_pred_br)/2
lr_br= (y_pred_br+y_pred_lr)/2
actual = {
    "Actual":y_test,
    "lrxg":lr_xg,
    "rflr":rf_lr,
    "lrgbr":lr_gbr,
    "brgbr":br_gbr,
    "etrbr":etr_br,
    "lrbr":lr_br,
}
actd = pd.DataFrame(actual)
actd.to_excel("pred.xlsx")
r2_lr_xg = r2_score(y_test,lr_xg)
r2_lr_rf = r2_score(y_test,rf_lr)
r2_lr_gbr = r2_score(y_test,lr_gbr)
r2_br_gbr = r2_score(y_test,br_gbr)
r2_br_etr = r2_score(y_test,etr_br)
r2_lr_br = r2_score(y_test,lr_br)
mae_lr_xg = mean_absolute_error(y_test,lr_xg)
mae_lr_rf = mean_absolute_error(y_test,rf_lr)
mae_lr_gbr =mean_absolute_error(y_test,lr_gbr)
mae_br_gbr =mean_absolute_error(y_test,br_gbr)
mae_br_etr =mean_absolute_error(y_test,etr_br)
mae_lr_br = mean_absolute_error(y_test,lr_br)
rmse_lr_xg= np.sqrt(mean_squared_error(y_test,lr_xg))
print("MSE: ",mean_squared_error(y_test,lr_xg))
rmse_lr_rf= np.sqrt(mean_squared_error(y_test,rf_lr))
rmse_lr_gbr= np.sqrt(mean_squared_error(y_test,lr_gbr))
rmse_br_gbr= np.sqrt(mean_squared_error(y_test,br_gbr))
rmse_lr_br= np.sqrt(mean_squared_error(y_test,lr_br))
rmse_br_etr= np.sqrt(mean_squared_error(y_test,etr_br))
rrse_lr_xg = rmse_lr_xg/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rrse_lr_rf = rmse_lr_rf/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rrse_lr_gbr =rmse_lr_gbr/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rrse_br_gbr =rmse_br_gbr/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rrse_br_etr =rmse_lr_xg/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rrse_lr_br = rmse_lr_xg/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))

rae_lr_xg = mae_lr_xg/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rae_lr_rf = mae_lr_rf/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rae_lr_gbr =mae_lr_gbr/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rae_br_gbr =mae_br_gbr/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rae_br_etr =mae_br_gbr/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rae_lr_br = mae_lr_br/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))


vaf_lr_xg=100 * (1 - (np.var(y_test - lr_xg) / np.var(y_test)))
vaf_lrrf=100 * (1 - (np.var(y_test - rf_lr) / np.var(y_test)))
vaf_lrgbr=100 * (1 - (np.var(y_test - lr_gbr) / np.var(y_test)))
vaf_br_gbr=100 * (1 - (np.var(y_test - br_gbr) / np.var(y_test)))
vaf_bretr=100 * (1 - (np.var(y_test - etr_br) / np.var(y_test)))
vaf_lrbr=100 * (1 - (np.var(y_test - lr_br) / np.var(y_test)))


# nse_lr = 1 - ( np.sum((Y_test - y_pred_knn)**2) / np.sum((Y_test - mean_observed)**2))
# print("NSE: ",nse_lr)

kge_lrxg = 1 - np.sqrt((np.corrcoef(y_test, lr_xg)[0, 1]- 1)**2 + (np.std(lr_xg) / np.std(y_test) - 1)**2 + (np.mean(lr_xg) / np.mean(y_test) - 1)**2)
kge_lrrf = 1 - np.sqrt((np.corrcoef(y_test, rf_lr)[0, 1]- 1)**2 + (np.std(rf_lr) / np.std(y_test) - 1)**2 + (np.mean(rf_lr) / np.mean(y_test) - 1)**2)
kge_lrgbr = 1 - np.sqrt((np.corrcoef(y_test, lr_gbr)[0, 1]- 1)**2 + (np.std(lr_gbr) / np.std(y_test) - 1)**2 + (np.mean(lr_gbr) / np.mean(y_test) - 1)**2)
kge_brgbr= 1 - np.sqrt((np.corrcoef(y_test, br_gbr)[0, 1]- 1)**2 + (np.std(br_gbr) / np.std(y_test) - 1)**2 + (np.mean(br_gbr) / np.mean(y_test) - 1)**2)
kge_bretr= 1 - np.sqrt((np.corrcoef(y_test, etr_br)[0, 1]- 1)**2 + (np.std(etr_br) / np.std(y_test) - 1)**2 + (np.mean(etr_br) / np.mean(y_test) - 1)**2)
kge_lrbr= 1 - np.sqrt((np.corrcoef(y_test, lr_br)[0, 1]- 1)**2 + (np.std(lr_br) / np.std(y_test) - 1)**2 + (np.mean(lr_br) / np.mean(y_test) - 1)**2)







fa= {
    "models":["lr&xg","lr&rf","lr&gbr","br&gbr","br&etr","lr&br"],
    "r2":[r2_lr_xg,r2_lr_rf,r2_lr_gbr,r2_br_gbr,r2_br_etr,r2_lr_br],
    "rmse":[rmse_lr_xg,rmse_lr_rf,rmse_lr_gbr,rmse_br_gbr,rmse_br_etr,rmse_lr_br],
    "mae":[mae_lr_xg,mae_lr_rf,mae_lr_gbr,mae_br_gbr,mae_br_etr,mae_lr_br],
    "rrse":[rrse_lr_xg,rrse_lr_rf,rrse_lr_gbr,rrse_br_gbr,rrse_br_etr,rrse_lr_br],
    "rae":[rae_lr_xg,rae_lr_rf,rae_lr_gbr,rae_br_gbr,rae_br_etr,rae_lr_br],
    "vaf":[vaf_lr_xg,vaf_lrrf,vaf_lrgbr,vaf_br_gbr,vaf_bretr,vaf_lrbr],
    "kge":[kge_lrxg,kge_lrrf, kge_lrgbr,kge_brgbr,kge_bretr,kge_lrbr],
    "nse":[r2_lr_xg,r2_lr_rf,r2_lr_gbr,r2_br_gbr,r2_br_etr,r2_lr_br]

}
fa = pd.DataFrame(fa)
fa.to_excel("comb_results1.xlsx")
print(fa)

# import pickle
# pickle.dump(rf_model,open('./others/rf_saved','wb'))
# loaded_rf = pickle.load(open('./others/rf_saved','rb'))

# prd= loaded_rf.predict(X_test)

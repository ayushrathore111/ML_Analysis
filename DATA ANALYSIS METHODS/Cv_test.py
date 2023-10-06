from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
import xgboost as xgb
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
import numpy as np
data= pd.read_excel("my data1.xlsx")
X= data.iloc[:,:-1]
y= data.iloc[:,-1]
# Assuming X and y are your feature and target matrices

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=None)
lr_model = LinearRegression()
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
knn_model = KNeighborsRegressor(n_neighbors=5)
etr_model = ExtraTreesRegressor(random_state=42)
dtr_model = DecisionTreeRegressor(random_state=42)
adaboost_model = AdaBoostRegressor(random_state=42)
bagging_model = BaggingRegressor(random_state=42)
xgb_model = xgb.XGBRegressor(random_state=42)

models = [knn_model,etr_model, dtr_model, adaboost_model, bagging_model, xgb_model,lr_model,rf_model,gbr_model]
print("testing CV: ")

y_pred_knn= knn_model.fit(X_train,y_train).predict(X_test)
y_pred_etr= etr_model.fit(X_train,y_train).predict(X_test)
y_pred_dtr= dtr_model.fit(X_train,y_train).predict(X_test)
y_pred_br= bagging_model.fit(X_train,y_train).predict(X_test)
y_pred_ar= adaboost_model.fit(X_train,y_train).predict(X_test)
y_pred_xg= xgb_model.fit(X_train,y_train).predict(X_test)
y_pred_lr= lr_model.fit(X_train,y_train).predict(X_test)
y_pred_rf= rf_model.fit(X_train,y_train).predict(X_test)
y_pred_gbr= gbr_model.fit(X_train,y_train).predict(X_test)
rmse_knn= np.sqrt(mean_squared_error(y_test,y_pred_knn))
rmse_dtr= np.sqrt(mean_squared_error(y_test,y_pred_dtr))
rmse_xg= np.sqrt(mean_squared_error(y_test,y_pred_xg))
rmse_etr= np.sqrt(mean_squared_error(y_test,y_pred_etr))
rmse_br= np.sqrt(mean_squared_error(y_test,y_pred_br))
rmse_ar= np.sqrt(mean_squared_error(y_test,y_pred_ar))
rmse_lr= np.sqrt(mean_squared_error(y_test,y_pred_lr))
rmse_rf= np.sqrt(mean_squared_error(y_test,y_pred_rf))
rmse_gbr= np.sqrt(mean_squared_error(y_test,y_pred_gbr))
dataf= {
    "models":["knn","etr","dtr","br","ar","xg","lr","rf","gbr"],
    "CV_RMSE":[rmse_knn , rmse_etr,rmse_dtr,rmse_br,rmse_ar
            ,rmse_xg,rmse_lr,rmse_rf,rmse_gbr]
}
dc= pd.DataFrame(dataf)
dc.to_excel("CV_testing.xlsx")
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model.__class__.__name__} - Root Mean Squared Error: {np.sqrt(mse)}")

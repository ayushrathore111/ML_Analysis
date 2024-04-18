from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression  # You can replace this with your preferred model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
import xgboost as xgb
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
import numpy as np
data= pd.read_excel("topic39_fcc.xlsx")
data= data.dropna()
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
from sklearn.svm import SVR
svr_model= SVR(kernel="linear", C=100, gamma=0.1, epsilon=0.1)
models = [knn_model,etr_model, bagging_model, adaboost_model, xgb_model,lr_model,rf_model,dtr_model,gbr_model,svr_model]
# Create KFold cross-validation with n_splits=10

kf = KFold(n_splits=10, shuffle=True, random_state=42)
res=[]

for model in models:
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)
    for i, r2 in enumerate(r2_scores):
        print(f'{model} Fold {i+1}: R^2 = {r2:.4f}')
    mean_r2 = r2_scores.mean()
    res.append(mean_r2)
    print(f'{model} Mean R^2 across all folds: {mean_r2:.4f}')

res = pd.DataFrame(res)
res.to_excel("cv_results.xlsx")























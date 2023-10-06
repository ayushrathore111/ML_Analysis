from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
import xgboost as xgb
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
import numpy as np
data= pd.read_excel("y2.xlsx")
X= data.iloc[:,:-1]
y= data.iloc[:,-1]
# Assuming X and y are your feature and target matrices

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Initialize models
etr_model = ExtraTreesRegressor(random_state=42)
dtr_model = DecisionTreeRegressor(random_state=42)
adaboost_model = AdaBoostRegressor(random_state=42)
bagging_model = BaggingRegressor(random_state=42)
xgb_model = xgb.XGBRegressor(random_state=42)

models = [etr_model, dtr_model, adaboost_model, bagging_model, xgb_model]

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model.__class__.__name__} - Root Mean Squared Error: {np.sqrt(mse)}")

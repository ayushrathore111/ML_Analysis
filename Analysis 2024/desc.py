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
data = pd.read_excel('topic27.xlsx')
from sklearn.neighbors import KNeighborsRegressor
# Separate the data into input X and output Y
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
print(y_test)
df = pd.DataFrame(X_train)
df_o=  pd.DataFrame(y_train)
df.to_excel("training_input.xlsx")
df_o.to_excel("training_ou.xlsx")
dy = pd.DataFrame(X_test)
dy_o = pd.DataFrame(y_test)
dy.to_excel("testing_in.xlsx")
dy_o.to_excel("testing_ou.xlsx")
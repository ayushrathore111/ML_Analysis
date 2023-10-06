from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble         import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network   import MLPRegressor
from numpy import mean
from numpy import absolute
from numpy import sqrt
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
data = pd.read_excel('load.xlsx')

# Separate the data into input X and output Y
data=data.dropna()
X = data.iloc[:, :-1]
y= data.iloc[:, -1]
cv = KFold(n_splits=10, random_state=0,shuffle=True)

#build multiple linear regression model
model0 = DecisionTreeRegressor()
model1 = LinearRegression()
model2= RandomForestRegressor()
model3 = GradientBoostingRegressor()
model4 = MLPRegressor()
model5 = KNeighborsRegressor()
model6 = ExtraTreesRegressor()


#use k-fold CV to evaluate model
scores0 = cross_val_score(model0, X, y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
scores1 = cross_val_score(model1, X, y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
scores2 = cross_val_score(model2, X, y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
scores3 = cross_val_score(model3, X, y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
scores4 = cross_val_score(model4, X, y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
scores5 = cross_val_score(model5, X, y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
scores6 = cross_val_score(model6, X, y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

#view mean absolute error
print("Decision Tree: ",mean(absolute(scores0)))
print("Linear regression: ",mean(absolute(scores1)))
print("Random Forest: ",mean(absolute(scores2)))
print("MLp: ",mean(absolute(scores3)))
print("GBR: ",mean(absolute(scores4)))
print("KNN: ",mean(absolute(scores5)))
print("etr: ",mean(absolute(scores6)))
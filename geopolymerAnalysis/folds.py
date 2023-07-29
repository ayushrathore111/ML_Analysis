from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from numpy import mean
from numpy import absolute
from numpy import sqrt
import pandas as pd
import numpy as np
data = pd.read_excel('bothgreter10.xlsx.xlsx')

# Separate the data into input X and output Y
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


#define cross-validation method to use
cv = KFold(n_splits=10, random_state=1, shuffle=True)

#build multiple linear regression model
model = LinearRegression()

#use k-fold CV to evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
print(sqrt(mean(absolute(scores))))
# accuracy


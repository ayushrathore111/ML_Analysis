import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Read data from Excel file
data = pd.read_excel('ecc.xlsx')

# Separate input features (X) and output values (y)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

etr_regressor = ExtraTreesRegressor()

grid_search = GridSearchCV(etr_regressor, param_grid, cv=5)
grid_search.fit(X_train, y_train) 

best_params = grid_search.best_params_
best_model = ExtraTreesRegressor(**best_params)
best_model.fit(X_train, y_train)
accuracy = best_model.score(X_test, y_test)
print("etr:", accuracy)

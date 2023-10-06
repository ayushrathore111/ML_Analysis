import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Read data from Excel file
data = pd.read_excel('load.xlsx')

# Separate input features (X) and output values (y)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

dt_regressor = DecisionTreeRegressor()

grid_search = GridSearchCV(dt_regressor, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = DecisionTreeRegressor(**best_params)
best_model.fit(X_train, y_train)
accuracy = best_model.score(X_test, y_test)
print("dtr:", accuracy)
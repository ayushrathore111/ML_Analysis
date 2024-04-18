import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor,ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
data = pd.read_excel('topic39.xlsx')

# Split data into features and target variable
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_test.to_excel('actual.xlsx')
# Preprocessing pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler()),  # Standardize numerical features
    ('encoder', OneHotEncoder())   # One-hot encode categorical features
])

# Fit preprocessing pipeline on training data and transform both training and testing data
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Define models and hyperparameters to search over
models_to_search = {
    'Random Forest Regressor': (RandomForestRegressor(), {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}),
    'Gradient Boosting Regressor': (GradientBoostingRegressor(), {'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.1, 0.2]}),
    'Extra Trees Regressor': (ExtraTreesRegressor(), {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}),
    'AdaBoost Regressor': (AdaBoostRegressor(), {'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.1, 0.2]}),
    'Bagging Regressor': (BaggingRegressor(), {'n_estimators': [10, 50, 100]}),
    'KNeighbors Regressor': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]}),
    'Linear Regression': (LinearRegression(), {}),
    'XGBoost Regressor': (XGBRegressor(), {'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.1, 0.2]}),
    'Decision Tree Regressor': (DecisionTreeRegressor(), {'max_depth': [None, 5, 10]}),
    'SVR': (SVR(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']})
}
excel_writer = pd.ExcelWriter('predictions_singlyModels.xlsx', engine='openpyxl')
# Grid search over models and hyperparameters
results = {}
for model_name, (model, param_grid) in models_to_search.items():
    clf = GridSearchCV(model, param_grid, cv=5)
    clf.fit(X_train_prep, y_train)
    results[model_name] = {
        'Best Parameters': clf.best_params_,
        'Train Score': clf.best_score_,
        'Test Score': clf.score(X_test_prep, y_test)
    }
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test_prep)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rrse = np.sqrt(mse) /(max(y_test.max(),y_train.max()) - min(y_test.min(),y_train.min()))
    rae = mae / (max(y_test.max(),y_train.max()) - min(y_test.min(),y_train.min()))
    VAF = 100 * (1 - (np.var(y_test - y_pred) / np.var(y_test)))
    r = np.corrcoef(y_test, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_test)
    beta = np.mean(y_pred) / np.mean(y_test)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    results_df = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred,'r2':r2,'rmse':rmse,'mse':mse,'mae':mae,'rrse':rrse,'rae':rae,'vaf':VAF,'kge':kge})
    results_df.to_excel(excel_writer, sheet_name=model_name, index=False)

# Print results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  Best Parameters: {metrics['Best Parameters']}")
    print(f"  Train Score: {metrics['Train Score']:.3f}")
    print(f"  Test Score: {metrics['Test Score']:.3f}")
    
excel_writer.save()
excel_writer.close()

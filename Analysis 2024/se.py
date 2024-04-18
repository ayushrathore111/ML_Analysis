import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

# Assuming you have your data and labels, replace X and y with your actual data
import pandas as pd
# Load the data with actual and predicted values
data = pd.read_excel("DEF FULL RCA.xlsx")
data = data.dropna()
X= data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Extra Trees model
extra_trees_model = KNeighborsRegressor()
extra_trees_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = extra_trees_model.predict(X_test)

# Calculate sensitivity (recall)
# sensitivity = recall_score(y_test, y_pred)
# print("Sensitivity (Recall): {:.4f}".format(sensitivity))

# Feature importance
feature_importance = extra_trees_model.feature_importances_
print(feature_importance)
dc = pd.DataFrame(feature_importance)
dc.to_excel("hello.xlsx")
# Sort features based on importance
sorted_indices = np.argsort(feature_importance)[::-1]

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), feature_importance[sorted_indices], align="center")
plt.xticks(range(X_train.shape[1]), np.array(X.columns)[sorted_indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Extra Trees - Feature Importance")
# plt.show()

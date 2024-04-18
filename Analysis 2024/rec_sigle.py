import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Load the data with actual and predicted values
data = pd.read_excel("topic39.xlsx")
data = data.dropna()
X= data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Decision Tree
DT = DecisionTreeRegressor(random_state=0)
DT.fit(X_train, Y_train)
DTR_Pred = DT.predict(X_test)

extra_trees_model = ExtraTreesRegressor()
extra_trees_model.fit(X_train, Y_train)
ETR_Pred = extra_trees_model.predict(X_test)

bagging_model = BaggingRegressor()
bagging_model.fit(X_train, Y_train)
BAGGING_Pred = bagging_model.predict(X_test)

model = XGBRegressor()
model.fit(X_train, Y_train)

adaboost_model = AdaBoostRegressor()
adaboost_model.fit(X_train, Y_train)
ar_Pred = adaboost_model.predict(X_test)

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train,Y_train)
knn_Pred = knn.predict(X_test)

lr = LinearRegression()
lr.fit(X_train,Y_train)
LR_Pred = lr.predict(X_test)

gbr = GradientBoostingRegressor()
rf = RandomForestRegressor()
gbr.fit(X_train,Y_train)
rf.fit(X_train,Y_train)
gbr_predict = gbr.predict(X_test)
rf_predict = rf.predict(X_test)

# Extract actual values
actual_values = data.iloc[:,-1]

# Extract model names
model_names = ["BR","AR","XGB","LR","RF","GBR"]  # Modify with your model names

# Initialize dictionaries to store accuracy and deviation
accuracy = {}
deviation = {}

# Calculate absolute errors
absolute_errors = {}
# absolute_errors["KNN"] = np.abs(knn_Pred - Y_test)
absolute_errors["BR"] = np.abs(BAGGING_Pred - Y_test)
absolute_errors["AR"]= np.abs(ar_Pred-Y_test)
absolute_errors["XGB"] = np.abs(model.predict(X_test) - Y_test)
absolute_errors["LR"] = np.abs(LR_Pred - Y_test)
absolute_errors["RF"] = np.abs(rf_predict - Y_test)
absolute_errors["GBR"] = np.abs(gbr_predict - Y_test)

# Calculate accuracy and deviation
for model, errors in absolute_errors.items():
    accuracy[model] = 1 - (errors / actual_values)
    deviation[model] = 1 - (errors / errors.mean())

# Calculate REC curve points
rec_curve_points = {}
for model, accuracy_values in accuracy.items():
    rec_curve_points[model] = []
    for threshold in np.linspace(0, 1, 101):
        rec_curve_points[model].append(np.mean(accuracy_values >= threshold))

# Plot REC curve with accuracy and deviation for each model
plt.figure(dpi=200)
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 12
# Set the figsize parameter for clearer and more visible plots
for model in model_names:
    plt.plot(np.linspace(0, 1, 101), rec_curve_points[model], marker='o', label=f"{model} REC Curve")
plt.xlabel("Threshold")
plt.ylabel("Metric Value")
plt.title("Comparison of REC Curves with Accuracy and Deviation")
plt.legend()
plt.grid(True)
plt.show()

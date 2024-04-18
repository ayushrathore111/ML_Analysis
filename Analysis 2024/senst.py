# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Load your dataset
# data = pd.read_excel('DEF FULL RCA.xlsx')
# data = data.dropna()
# # Split the dataset into training and testing sets
# X = data.iloc[:,:-1]
# y = data.iloc[:,-1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# # Train a machine learning model (Linear Regression in this example)
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Calculate the baseline mean squared error
# baseline_mse = mean_squared_error(y_test, y_pred)

# # Perform sensitivity analysis by perturbing one feature at a time
# sensitivity_results = {}

# for feature in X.columns:
#     perturbed_X_test = X_test.copy()
    
#     # Add some perturbation to the feature
#     perturbation_amount = 0.01  # Adjust the perturbation amount as needed
#     perturbed_X_test[feature] += perturbation_amount
    
#     # Make predictions with the perturbed feature
#     perturbed_y_pred = model.predict(perturbed_X_test)
    
#     # Calculate the mean squared error with the perturbed feature
#     perturbed_mse = mean_squared_error(y_test, perturbed_y_pred)
    
#     # Calculate sensitivity as the change in mean squared error
#     sensitivity = max(0,perturbed_mse - baseline_mse)
    
#     sensitivity_results[feature] = sensitivity

# # Display sensitivity results
# sensotivity=[]
# features=[]

# for feature, sensitivity in sensitivity_results.items():
#     sensotivity.append(sensitivity)
#     features.append(feature)
#     print(f'Sensitivity of {feature}: {sensitivity}')

# dc = pd.DataFrame(sensotivity,features)
# dc.to_excel("sne.xlsx")
# # You can analyze the sensitivity results to understand which features have the most impact on the model's predictions.


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
import numpy as np
data= pd.read_excel("DEF FULL RCA.xlsx")
data= data.dropna()
X= data.iloc[:,:-1]
y= data.iloc[:,-1]
# Assuming you have your data and labels, replace X and y with your actual data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_regression_model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: {:.4f}".format(mse))

# Sensitivity analysis for linear regression is often done by analyzing coefficients
coefficients = linear_regression_model.coef_
feature_names = X.columns

dc = pd.DataFrame(coefficients)
dc.to_excel("sne.xlsx")

# Plotting coefficients
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_names)), coefficients)
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Linear Regression - Coefficient Importance")
plt.show()

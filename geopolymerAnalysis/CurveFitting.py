import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Read the data from CSV
data = pd.read_excel('fly_ash.xlsx')
data=data.dropna()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Perform curve fitting for each feature individually
curve_params = []
curve_covariance = []

for column in X.columns:
    x_data = X[column].values
    y_data = y.values

    # Define the model function
    def model_function(x, a, b, c):
        return a * np.sin(b * x) + c

    # Perform the curve fitting
    initial_guess = [1, 1, 1]  # Initial parameter guess for the model function
    fit_params, fit_covariance = curve_fit(model_function, x_data, y_data, p0=initial_guess)

    curve_params.append(fit_params)
    curve_covariance.append(fit_covariance)

# Plot the original data and the fitted curves
plt.figure(figsize=(12, 8))

for i, column in enumerate(X.columns):
    x_data = X[column].values
    y_data = y.values

    a_opt, b_opt, c_opt = curve_params[i]
    y_fit = model_function(x_data, a_opt, b_opt, c_opt)

    plt.scatter(x_data, y_data, label='Data - ' + column)
    plt.plot(x_data, y_fit, label='Fitted Curve - ' + column)

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Fitting Example')
plt.show()

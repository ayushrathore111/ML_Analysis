import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Split the dataset into training and testing sets
X = data[:,:-1]
y = data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train a machine learning model (Linear Regression in this example)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the baseline mean squared error
baseline_mse = mean_squared_error(y_test, y_pred)

# Perform sensitivity analysis by perturbing one feature at a time
sensitivity_results = {}

for feature in X.columns:
    perturbed_X_test = X_test.copy()
    
    # Add some perturbation to the feature
    perturbation_amount = 0.1  # Adjust the perturbation amount as needed
    perturbed_X_test[feature] += perturbation_amount
    
    # Make predictions with the perturbed feature
    perturbed_y_pred = model.predict(perturbed_X_test)
    
    # Calculate the mean squared error with the perturbed feature
    perturbed_mse = mean_squared_error(y_test, perturbed_y_pred)
    
    # Calculate sensitivity as the change in mean squared error
    sensitivity = perturbed_mse - baseline_mse
    
    sensitivity_results[feature] = sensitivity

# Display sensitivity results
for feature, sensitivity in sensitivity_results.items():
    print(f'Sensitivity of {feature}: {sensitivity}')

# You can analyze the sensitivity results to understand which features have the most impact on the model's predictions.

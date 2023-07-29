import pandas as pd
import numpy as np

class SANN:
    def __init__(self, input_dim, output_dim, learning_rate=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        self.weights = np.random.randn(input_dim, output_dim)
    
    def predict(self, x):
        return np.dot(x, self.weights)
    
    def train(self, x):
        y = self.predict(x)
        winner_index = np.argmax(y)
        
        # Update the weights of the winner neuron
        self.weights[:, winner_index] += self.learning_rate * (x - self.weights[:, winner_index])

# Read the data from the Excel file
data = pd.read_excel('EWS M.xlsx')

# Extract the 'discharge' column for training
discharge_data = data['discharge'].values

# Normalize the data
discharge_data = (discharge_data - np.mean(discharge_data)) / np.std(discharge_data)

# Define the input and output dimensions
input_dim = 1
output_dim = 10  # Set the number of output neurons as needed

# Create an instance of SANN
model = SANN(input_dim, output_dim)

# Training loop
for i in range(len(discharge_data)):
    # Retrieve the current data point
    x = np.array([discharge_data[i]])
    
    # Train the SANN model
    model.train(x)
    
    # Print the predicted output
    print("Epoch:", i)
    print("Predicted output:", model.predict(x))

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
y_true = discharge_data[input_dim:]  # Assuming you are predicting one timestep ahead
y_pred = [model.predict(np.array([discharge_data[i]])) for i in range(input_dim, len(discharge_data))]

r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

# Print the evaluation metrics
print("R2 Score:", r2)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
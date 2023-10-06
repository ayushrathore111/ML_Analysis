##FFNN(Multilayer Feedforward Neural Network)
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import pandas as pd

# Load data from Excel
data = pd.read_excel('y2.xlsx')

# Separate the data into input X and output y
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Build the FFNN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(X_train.shape[1],)),  # Use the number of input features
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model....
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model.....
model.fit(X_train, Y_train, epochs=100, batch_size=8, verbose=0)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))

# Calculate R2 score

r2 = r2_score(Y_test, y_pred)
rmse = mean_squared_error(Y_test, y_pred, squared=False)
mse = mean_squared_error(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
rrse= np.sqrt(mse) / (Y_test.max() - Y_test.min())
rae = mae / (Y_test.max() - Y_test.min())

print("r2: ",r2)
print("rmse: ",rmse)
print("mse: ",mse)
print("mae: ",mae)
print("rrse: ",rrse)
print("rae: ",rae)

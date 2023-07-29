import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# Read the data from a file
data = pd.read_excel('EWS M.xlsx')

# Convert the datetime column to a compatible data type (e.g., Unix timestamps)
data['Month'] = pd.to_datetime(data['Month'])
data['Month'] = data['Month'].apply(lambda x: x.timestamp())

# Convert the data to NumPy arrays
X = np.array(data['Month'])
y = np.array(data['discharge'])

# Proceed with the rest of your TensorFlow model code...

# Normalize the target values
scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build the ANN model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(1,)))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(loss)
# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions to get original scale
dc = pd.DataFrame(predictions)
dc.to_excel("ann_predictions.xlsx")
predictions = scaler.inverse_transform(predictions)
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)

# Print the evaluation metrics
print("R2 Score:", r2)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
pd.DataFrame(y_test).to_excel("ytest.xlsx")
# pd.DataFrame(X_test).to_excel("xtest1.xlsx")

# Perform further analysis or visualization

#fnn sann tlnn
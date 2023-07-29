import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Read the data from a file
data = pd.read_excel('EWS M.xlsx')

# Convert the datetime column to a compatible data type (e.g., Unix timestamps)
data['Month'] = pd.to_datetime(data['Month'])
data['Month'] = data['Month'].apply(lambda x: x.timestamp())

# Convert the data to NumPy arrays
X = np.array(data['Month'])
y = np.array(data['discharge'])

# Normalize the target values
scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Reshape the input data
X_train = X_train.reshape(-1, 1, 1)
X_test = X_test.reshape(-1, 1, 1)

# Define the RNN model
model = Sequential()
model.add(LSTM(128, input_shape=(1, 1)))  # Add an LSTM layer with 128 units
model.add(Dense(1))  # Add a fully connected layer with the desired output dimension

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100)

# Make predictions
predictions = model.predict(X_test)
dc = pd.DataFrame(predictions)
dc.to_excel("tlnn_predictions.xlsx")
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)

# Print the evaluation metrics
print("------tlnn----------")
print("R2 Score:", r2)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)











# import pandas as pd
# import numpy as np
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,LSTM

# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# # Read the data from a file
# data = pd.read_excel('EWS M.xlsx')

# # Convert the datetime column to a compatible data type (e.g., Unix timestamps)
# data['Month'] = pd.to_datetime(data['Month'])
# data['Month'] = data['Month'].apply(lambda x: x.timestamp())

# # Convert the data to NumPy arrays
# X = np.array(data['Month'])
# y = np.array(data['discharge'])

# # Proceed with the rest of your TensorFlow model code...

# # Normalize the target values
# scaler = MinMaxScaler()
# y = scaler.fit_transform(y.reshape(-1, 1))

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # Define the RNN model
# model = Sequential()
# model.add(LSTM(128, input_shape=(1,)))  # Add an LSTM layer with 128 units
# model.add(Dense(1))  # Add a fully connected layer with the desired output dimension

# # Compile the model
# model.compile(loss='mean_squared_error', optimizer='adam')
# print(X_train.shape)
# print(y_train.shape)
# # Train the model
# model.fit(X_train, y_train, epochs=100)

# # Make predictions
# predictions = model.predict(X_test)
# dc = pd.DataFrame(predictions)
# dc.to_excel("tlnn_predictions.xlsx")
# r2 = r2_score(y_test, predictions)
# mse = mean_squared_error(y_test, predictions)
# mae = mean_absolute_error(y_test, predictions)
# rmse = np.sqrt(mse)

# # Print the evaluation metrics
# print("R2 Score:", r2)
# print("Mean Squared Error (MSE):", mse)
# print("Mean Absolute Error (MAE):", mae)
# print("Root Mean Squared Error (RMSE):", rmse)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load data from Excel file into a DataFrame
df = pd.read_excel('topic39.xlsx')

# Assuming your Excel file has columns for features and a column for the target variable
# Extract features and target variable from the DataFrame
X = df.iloc[:,:-1]  # Features
y = df.iloc[:,-1]  # Target variable

# Convert DataFrame to numpy arrays
X = np.array(X)
y = np.array(y)

# Define the fully connected neural network model
def create_model(input_shape, num_classes):
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        # Flatten layer
        layers.Flatten(),
        # Dense (fully connected) layers
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Define input shape and number of classes
input_shape = X.shape[1]  # Number of features
num_classes = len(np.unique(y))  # Number of unique classes in your target variable

# Create the fully connected neural network model
model = create_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

##RBNN(Radius basis neural network)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from scipy.spatial.distance import cdist

class RBFNN:
    def __init__(self, n_centers, learning_rate=0.01):
        self.n_centers = n_centers
        self.learning_rate = learning_rate
        self.centers = None
        self.widths = None
        self.weights = None

    def gaussian_kernel(self, X, center, width):
        return np.exp(-width * np.linalg.norm(X - center)**2)

    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.n_centers, random_state=0).fit(X)
        self.centers = kmeans.cluster_centers_

        pairwise_dists = cdist(X, self.centers)
        self.widths = np.std(pairwise_dists)

        self.weights = np.random.randn(self.n_centers)
        for epoch in range(100):
            for i in range(X.shape[0]):
                phi = np.array([self.gaussian_kernel(X[i], c, self.widths) for c in self.centers])
                prediction = np.dot(phi, self.weights)
                error = y[i] - prediction
                self.weights += self.learning_rate * error * phi

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            phi = np.array([self.gaussian_kernel(X[i], c, self.widths) for c in self.centers])
            prediction = np.dot(phi, self.weights)
            predictions.append(prediction)
        return np.array(predictions)
import pandas as pd
data = pd.read_excel('load.xlsx')

# Separate the data into input X and output Y
# data=data.dropna()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create and train the RBFNN
# Convert X DataFrame to a NumPy array
X_train_array = X_train.values
y_train_array = y_train.values

# Create and train the RBFNN
rbfnn = RBFNN(n_centers=15, learning_rate=0.1)
rbfnn.fit(X_train_array, y_train_array)

# Make predictions
y_pred = rbfnn.predict(X_test.values)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
r2 = r2_score(y_pred, y_test)
print("r2:", r2)
print("rmse:", rmse)
VAF = 100 * (1 - (np.var(y_test - y_pred) / np.var(y_test)))
print("VAF: ",VAF)


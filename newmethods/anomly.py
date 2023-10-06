from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import pandas as pd
data = pd.read_excel('load.xlsx')
# Example data
X = data.iloc[:,:-1]
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN for clustering-based anomaly detection
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Isolation Forest for isolation-based anomaly detection
iso_forest = IsolationForest(contamination=0.05)
isolation_scores = iso_forest.fit_predict(X_scaled)

# Principal Component Analysis (PCA) for anomaly detection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(X_pca)
print(pca)

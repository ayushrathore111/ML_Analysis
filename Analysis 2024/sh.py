import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
boston = pd.read_excel('topic39_fcc.xlsx')
X = boston.iloc[:,:-1]
y = boston.iloc[:,-1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Bagging Regressor
bagging_reg = BaggingRegressor(n_estimators=100, random_state=42)
bagging_reg.fit(X_train_scaled, y_train)
def model_predictor(X):
    return bagging_reg.predict(X)

# SHAP explanation
explainer = shap.Explainer(model_predictor, X_train_scaled)
shap_values = explainer(X_test_scaled)

# SHAP explanation
# explainer = shap.Explainer(bagging_reg, X_train_scaled)
# shap_values = explainer(X_test_scaled)
plt.figure(dpi=250)
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 16 # 
# Summary plot
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)

# plt.savefig('shap_summary_plot.png', dpi=300)









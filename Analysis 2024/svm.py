import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,r2_score,mean_absolute_error,mean_squared_error

excel_file_path = '28-12-2023 Fatigue data ANN.xlsx'
df = pd.read_excel(excel_file_path)

# Assuming you have a column named 'target' that you want to predict
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Support Vector Machine model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # You can adjust the parameters based on your data

# Train the model
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

r2_lr_xg = r2_score(y_test,y_pred)
mae_lr_xg = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rrse_lr_xg = np.sqrt(mse)/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
rae_lr_xg = mae_lr_xg/(max(y_test.max(),y_train.max())-min(y_test.min(),y_train.min()))
vaf_lr_xg=100 * (1 - (np.var(y_test - y_pred) / np.var(y_test)))
kge_lrxg = 1 - np.sqrt((np.corrcoef(y_test, y_pred)[0, 1]- 1)**2 + (np.std(y_pred) / np.std(y_test) - 1)**2 + (np.mean(y_pred) / np.mean(y_test) - 1)**2)


print("R2: ",r2_lr_xg)
print("MSE: ",mse)
print("RMSE: ",np.sqrt(mse))
print("MAE: ",mae_lr_xg)
print("RRSE: ",rrse_lr_xg)
print("RAE: ",rae_lr_xg)
print("VAF: ",vaf_lr_xg)
print("KGE: ",kge_lrxg)

print(f'Accuracy: {accuracy}')

# Display classification report
print(classification_report(y_test, y_pred))

# Save the model (optional)
from joblib import dump
dump(svm_model, 'your_svm_model.joblib') 
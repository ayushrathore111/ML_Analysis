import shap
import pandas as pd
import matplotlib.pyplot as plt  # Import Matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

# Load your data from the file "data1" using pandas
data = pd.read_excel("y2.xlsx")  # Replace with the actual file name

# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train an ETR model (replace with your own configuration)
etr_model = ExtraTreesRegressor(n_estimators=100)
etr_model.fit(X_train, y_train)

# Create a SHAP explainer
explainer = shap.Explainer(etr_model)

# Explain predictions for the test set
shap_values = explainer(X_test)

# Configure Matplotlib to use "Times New Roman" font
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 12 # Adjust the font size as needed

# Plot SHAP summary plot
shap.summary_plot(shap_values, X_test)

# Optionally, save the plot to an image file
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
data = pd.read_excel('load.xlsx')
# Example data
data= data.dropna()
X = data.iloc[:,:-1]
y=data.iloc[:,-1]
# Univariate feature selection
selector = SelectKBest(score_func=f_classif, k=6)
X_selected = selector.fit_transform(X, y)

# Feature importance from a random forest
model = RandomForestClassifier()
model.fit(X, y)
feature_importances = model.feature_importances_
print(feature_importances)

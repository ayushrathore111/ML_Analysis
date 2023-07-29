
from sklearn.model_selection import train_test_split 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('both.csv')

X = data.iloc[:, :-1]
y= data.iloc[:,-1]


# digitizing continuous variable
aa = X['DAYS']
minima = aa.min()
maxima = aa.max()
bins = np.linspace(minima-1,maxima+1, 3)
binned = np.digitize(aa, bins)
plt.hist(binned, bins=50)
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,
                                        random_state=101,stratify=binned)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(X_train)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)
y_train_shaped = np.reshape(y_train,(-1,1))
y_test_shaped = np.reshape(y_test,(-1,1))
scaler_y.fit(y_train_shaped)
y_train = scaler_y.transform(y_train_shaped)
y_test = scaler_y.transform(y_test_shaped)


from ANFIS import EVOLUTIONARY_ANFIS

E_Anfis = EVOLUTIONARY_ANFIS(functions=3,generations=500,offsprings=10,
                            mutationRate=0.2,learningRate=0.2,chance=0.7,ruleComb="simple")

bestParam, bestModel = E_Anfis.fit(X_train,y_train,optimize_test_data=False)

bestParam, bestModel = E_Anfis.fit(X_train,y_train,X_test,y_test,optimize_test_data=True)

from scipy.stats import pearsonr
pred_train = E_Anfis.predict(X_train,bestParam,bestModel)
pearsonr(y_train,pred_train.reshape(-1,1))

pred_test = E_Anfis.predict(X_test,bestParam,bestModel)
pearsonr(y_test,pred_test.reshape(-1,1))
        
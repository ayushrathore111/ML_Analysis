import pandas as pd
import numpy as np
data = pd.read_csv('fly_ash.csv')
X = data.iloc[:, :-1]
y= data.iloc[:,-1]

print(X.columns)
#feature design
days = pd.get_dummies(X['DAYS'],drop_first=True)
ggbs = pd.get_dummies(X['NaOh(kg/m3)'],drop_first=True)
# print(ggbs)
X.drop(['DAYS','NaOh(kg/m3)'],axis=1)
X=pd.concat([X,days,ggbs],axis=1)
import tensorflow
opt=tensorflow.keras.optimizers.Adam(learning_rate=0.01)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU,ReLU
from tensorflow.keras.layers import Dropout

classifier = Sequential()
#input layer
classifier.add(Dense(units=11,activation='relu'))
#first layer
classifier.add(Dense(units=7,activation='relu'))
#second layer
classifier.add(Dense(units=6,activation='relu'))
#output layer
classifier.add(Dense(1,activation='sigmoid'))
classifier.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

model_history= classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=100)
print(model_history)



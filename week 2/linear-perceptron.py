# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#load train and test datasets into X,y 
df_train = pd.read_csv('perceptron-train.csv', header=None)
X_train = df_train.iloc[:,1:]
y_train = df_train[0]
df_test = pd.read_csv('perceptron-test.csv', header=None)
X_test = df_test.iloc[:,1:]
y_test = df_test[0]

print y_test
#iniitialize and fit perceptron model
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
#predict and compute accuracy
unscaled_accuracy = accuracy_score(y_test, clf.predict(X_test))

#scale all features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#fit on scaled data
clf.fit(X_train_scaled, y_train)
#compute new accuracy
scaled_accuracy = accuracy_score(y_test, clf.predict(X_test_scaled))

#how much are we doing better now?
print round(scaled_accuracy - unscaled_accuracy, 3)
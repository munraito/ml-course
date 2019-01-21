# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss

df = pd.read_csv('gbm-data.csv')
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

learn_rates = [1, 0.5, 0.3, 0.2, 0.1]

for lr in learn_rates:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, learning_rate=lr, random_state=241)
    clf.fit(X_train, y_train)
    #compute quality on training set
    train_loss = []
    for i, y_pred in enumerate(clf.staged_decision_function(X_train)):
        y_pred_sigmoid =  1.0 / (1 + np.exp(-y_pred))
        loss = log_loss(y_train, y_pred_sigmoid)
        train_loss.append(loss)
       
    #compute quality and find minimum loss on test set
    min_loss = [0, 10]
    test_loss = []
    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        y_pred_sigmoid =  1.0 / (1 + np.exp(-y_pred))
        loss = log_loss(y_test, y_pred_sigmoid)
        test_loss.append(loss)
        if loss < min_loss[1]:
            min_loss[0] = i
            min_loss[1] = loss
    
    print 'Learn rate = ', lr, 'Min loss: ', min_loss
    #plot losses on training set vs losses on test set
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(['test', 'train'])
    plt.show()

#Compare with Random Forest
clf = RandomForestClassifier(n_estimators=45, random_state=241)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
print log_loss(y_test, y_pred)
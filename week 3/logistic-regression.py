# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data-logistic.csv', header=None)
X = df.values[:, 1:]
y = df.values[:, 0]
print X[:,0].shape
print y.shape

def sigmoid(x):
    return 1.0 / (1 + np.exp(x))

def distance(a, b):
    return np.sqrt(np.square(a[0] - b[0]) + np.square(a[1] - b[1]))

def LogisticRegression(X, y, w, k, C, eps, max_iter):
    w1, w2 = w
    for i in range(max_iter):
        w1_new = w1 + k * np.mean(y * X[:,0] * (1 - 1./(1 + np.exp(-y*(w1*X[:,0] + w2*X[:,1]))))) - k*C*w1
        w2_new = w2 + k * np.mean(y * X[:,1] * (1 - 1./(1 + np.exp(-y*(w1*X[:,0] + w2*X[:,1]))))) - k*C*w2
        if distance((w1_new, w2_new), (w1, w2)) < eps:
            print i
            break
        w1, w2 = w1_new, w2_new
    
    predictions = []
    for i in range(len(X)):
        t1 = -w1*X[i, 0] - w2*X[i, 1]
        s = sigmoid(t1)
        predictions.append(s)
    return predictions

p0 = LogisticRegression(X, y, [0.0, 0.0], 0.1, 0, 0.00001, 10000)
p1 = LogisticRegression(X, y, [0.0, 0.0], 0.1, 10, 0.00001, 10000)

print (roc_auc_score(y, p0))
print (roc_auc_score(y, p1))
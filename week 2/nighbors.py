# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale

def find_best_neighbors(X,y):
    #initialize KFold object and list of integers we want to try
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    k_range = range(1,51)
    
    max_score = (1, 0)
    #iterate through the list and compute accuracy
    for k in k_range:
        kNN = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(kNN, X, y, cv=kf, scoring='accuracy')
        #find maximum accuracy score
        if (scores.mean() > max_score[1]):
            max_score = (k, scores.mean())
    return max_score

#read data
df = pd.read_csv('wine.data', header=None)

#extract features and target
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

#first iteration, unscaled features
print find_best_neighbors(X,y)
#scale features
X = scale(X)
#find accuracy of unscaled
print find_best_neighbors(X,y)
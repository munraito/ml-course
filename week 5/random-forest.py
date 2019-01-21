# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

#import and preprocess data
df = pd.read_csv('abalone.csv')
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

#define features and target
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

#initialize KFold object
kf = KFold(n_splits=5, random_state=1, shuffle=True)
trees_num = range(1,51)

#compute best number of trees in random forest
for i in trees_num:
    rf = RandomForestRegressor(n_estimators=i)
    scores = cross_val_score(rf, X, y, cv=kf,
                             scoring='r2')
    print 'num of trees = ', i, ', R2 score = ', scores.mean()
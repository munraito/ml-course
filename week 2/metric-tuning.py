# -*- coding: utf-8 -*-
from sklearn.datasets import load_boston
import numpy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale

#load Boston dataset
boston = load_boston()

#scale features
X = scale(boston.data)
y = boston.target

#generate list of power parameters for the model
power = numpy.linspace(1,10,num=50)

#5-fold object for cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
#variable for maximum accuracy
max_score = (1, -1000)

#iterate through the list and compute accuracy for each p
for p in power:
    model = KNeighborsRegressor(n_neighbors=5, weights='distance',
                                metric='minkowski', p=p)
    scores = cross_val_score(model, X, y, cv=kf,
                             scoring='neg_mean_squared_error')
    print scores
    if (scores.mean() > max_score[1]):
        max_score = (p, scores.mean())

print max_score
    
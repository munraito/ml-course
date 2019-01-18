# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

#import close prices and Dow Jones Industrial Average
prices = pd.read_csv('close_prices.csv')
X = prices.iloc[:,1:]
index = pd.read_csv('djia_index.csv')
DJI = index['^DJI']

#generate new feature matrix using Principal Component Analysis
pca = PCA(n_components=10)
new_X = pca.fit_transform(X)
#find how much components can explain 90& variance
print pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1] + pca.explained_variance_ratio_[2] + pca.explained_variance_ratio_[3]

#Pearson's correlation between first component and Dow Jones index
print np.corrcoef([new_X[:,0], DJI])

#find company with the biggest weight 
companies = prices.columns[1:]
first_comp = pca.components_[0]
m = max(first_comp)
max_index = [i for i, j in enumerate(first_comp) if j == m]

print companies[max_index]
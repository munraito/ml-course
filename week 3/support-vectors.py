# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.svm import SVC

#import data and take features / target
df = pd.read_csv('svm-data.csv', header=None)
X = df.iloc[:,1:]
y = df[0]

#initialize linear SVM, train it on data
clf = SVC(C=100000, random_state=241, kernel='linear')
clf.fit(X,y)
#get support indicies
print clf.support_ + 1
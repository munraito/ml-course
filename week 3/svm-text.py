# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC

#import news
newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )
#init RF-IDF object
vectorizer = TfidfVectorizer()
#fit it to the dataset and get target vector
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target
#get list of feature names
feature_mapping = vectorizer.get_feature_names()

#find best C for SVM Classiffier
"""
grid = {'C': np.power(10.0, np.arange(-5, 5))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

max_score = (0,)
for a in gs.grid_scores_:
    print a.mean_validation_score, a.parameters
    if a.mean_validation_score > max_score[0]:
        max_score = (a.mean_validation_score, a.parameters)
       
#print max_score
""" 
#best C = 1
#fit SVM with best C
clf = SVC(kernel='linear', random_state=241, C=1)
clf.fit(X,y)
#extract coefficients for every word and print top 10 words with high coefs
word_indexes = np.argsort(np.abs(clf.coef_.toarray()[0]))[-10:]
words = [feature_mapping[i] for i in word_indexes]
print ' '.join(sorted(words))

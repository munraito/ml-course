# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

#import train and test datasets, also define target
train = pd.read_csv('salary-train.csv')
test = pd.read_csv('salary-test-mini.csv')
y = train['SalaryNormalized']

#preprocess text data
train['FullDescription'] = train['FullDescription'].str.lower()
train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

test['FullDescription'] = test['FullDescription'].str.lower()
test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

#vectorize text
vectorizer = TfidfVectorizer(min_df=5)
train_vectorized = vectorizer.fit_transform(train['FullDescription'])
test_vectorized = vectorizer.transform(test['FullDescription'])

#fill NAs
train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)

#one-hot encoding of features
enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))

train_features_matrix = hstack([train_vectorized, X_train_categ])
test_features_matrix = hstack([test_vectorized, X_test_categ])

#fit and predict using Ridge regression
model = Ridge(alpha=1, random_state=241)
model.fit(train_features_matrix, y)

print model.predict(test_features_matrix)
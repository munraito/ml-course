# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('titanic.csv', index_col='PassengerId')

df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})

df1 = df[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna(axis=0)

X = df1[['Pclass', 'Fare', 'Age', 'Sex']]
y = df1['Survived']

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

print clf.feature_importances_
print dict(zip(X.columns, clf.feature_importances_))
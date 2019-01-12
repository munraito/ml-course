# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from collections import Counter
df = pd.read_csv('titanic.csv', index_col='PassengerId')

print df.info()

print 'Survived passengers rate: ', len(df[df['Survived'] == True]) / float(len(df['Survived']))
print 'First class passengers rate: ', len(df[df['Pclass'] == 1]) / float(len(df['Survived']))
print 'Age median: ', df[df.Age.notnull() == True]['Age'].median()
print 'Age mean: ', df[df.Age.notnull() == True]['Age'].mean()
print 'Correlation between SibSp and Parch: ', df['SibSp'].corr(df['Parch'], method='pearson')

Names = df[df['Sex'] == 'female']['Name']
FirstNames = []

for name in Names:
    m = re.search('\((.+?)(\s|\))', name)
    if m:
        FirstNames.append(m.group(1).strip())
    else:
        dot = re.search('\.(.+?)(\s|$)', name)
        if dot:
            FirstNames.append(dot.group(1).strip())

Counter = Counter(FirstNames)

print 'most popular female name: ', Counter.most_common(5)
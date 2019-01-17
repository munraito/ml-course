# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

df = pd.read_csv('classification.csv')
#confusion matrix:
print 'True Positive: ', df[(df.true == 1) & (df.pred == 1)].count()
print 'False Positive: ', df[(df.true == 0) & (df.pred == 1)].count()
print 'False Negative: ', df[(df.true == 1) & (df.pred == 0)].count()
print 'True Negative: ', df[(df.true == 0) & (df.pred == 0)].count()

#key metrics:
print 'Accuracy: ', accuracy_score(df.true, df.pred)
print 'Precision: ', precision_score(df.true, df.pred)
print 'Recall: ', recall_score(df.true, df.pred)
print 'F1: ', f1_score(df.true, df.pred)

df2 = pd.read_csv('scores.csv')
#ROC-AUC score
print 'LogReg score:', roc_auc_score(df2.true, df2.score_logreg)
print 'SVM score:', roc_auc_score(df2.true, df2.score_svm)
print 'KNN score:', roc_auc_score(df2.true, df2.score_knn)
print 'Tree score:', roc_auc_score(df2.true, df2.score_tree)

#PR curve:
precision, recall, _ = precision_recall_curve(df2.true, df2.score_logreg)
print 'Precision LogReg: ', max(precision[recall >= 0.7])
precision, recall, _ = precision_recall_curve(df2.true, df2.score_svm)
print 'Precision SVM: ', max(precision[recall >= 0.7])
precision, recall, _ = precision_recall_curve(df2.true, df2.score_knn)
print 'Precision KNN: ', max(precision[recall >= 0.7])
precision, recall, _ = precision_recall_curve(df2.true, df2.score_tree)
print 'Precision Tree: ', max(precision[recall >= 0.7])
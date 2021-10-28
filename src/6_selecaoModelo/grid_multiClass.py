#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from utils.EstimatorSelectionHelper import EstimatorSelectionHelper
from warnings import filterwarnings
from sklearn.model_selection._split import KFold
filterwarnings('ignore')

# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# https://medium.com/swlh/hyperparameter-tuning-of-support-vector-machine-using-gridsearchcv-4d17671d1ed2
# https://panjeh.medium.com/scikit-learn-hyperparameter-optimization-for-mlpclassifier-4d670413042b
# https://towardsdatascience.com/logistic-regression-model-tuning-with-scikit-learn-part-1-425142e01af5
def model_select(dataset):
    df = pd.read_csv('../5_reducaoDimensionalidade/datasets/{}_reduced_pca.csv'.format(dataset), sep=';', header=0)
    
    x = df.drop('_class_', axis=1)
    y = df['_class_']
    
    x_train, _, y_train, _ = train_test_split(x, y,
                                              test_size=.3,
                                              random_state=183212)
    
    cv = KFold(n_splits=10, shuffle=True)
    cv.get_n_splits(df)
    
    parametros = {'SVM':{'estimator__C':[0.1, 1, 10], 'estimator__gamma':[0.01, 0.1, 1]},
                  'RandomForestClassifier':{'estimator__n_estimators': np.arange(80, 100, 10), 'estimator__max_depth': np.arange(3, 5), 'estimator__criterion':['gini', 'entropy']},
                  'MLPClassifier':{'estimator__hidden_layer_sizes': [(10, 30, 10), (20,)], 'estimator__activation': ['tanh', 'relu'], 'estimator__solver': ['sgd', 'adam']},
                  'LogisticRegression':{'estimator__penalty': ['l1', 'l2'], 'estimator__C': [0.1, 1, 10]}
                  }

    models = {'SVM':OneVsRestClassifier(SVC(class_weight='balanced')),
              'RandomForestClassifier':OneVsRestClassifier(RandomForestClassifier(class_weight='balanced')),
              'MLPClassifier':OneVsRestClassifier(MLPClassifier()),
              'LogisticRegression':OneVsRestClassifier(LogisticRegression(class_weight='balanced'))
              }
    
    print(dataset)
    estimador = EstimatorSelectionHelper(models, parametros)
    estimador.fit(x_train, y_train, cv=cv, n_jobs=1, scoring='accuracy')
    estimador.best_model(dataset)

if __name__ == '__main__':
    
    datasets = ['tfidf-11','tfidf-12', 'bow-11', 'bow-12']

    for dataset in datasets:
        model_select(dataset)
    
    print("Fim da selecao dos hiperparemetros para multi-classes")
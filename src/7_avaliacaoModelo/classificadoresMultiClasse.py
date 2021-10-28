#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from utils.metricasAvaliacao import metricasAvaliacao
from os import path, makedirs
from warnings import filterwarnings
filterwarnings('ignore')

def svm(x_train, x_test, y_train, y_test, cv, output_filename):
    
    print(output_filename)    
    print('svm')
    
    params = {'estimator__C':[1,10], 
              'estimator__gamma':[0.01, 1]}
    
    clf = GridSearchCV(OneVsRestClassifier(SVC(class_weight='balanced')), params, cv=cv,
                       n_jobs=1, verbose=1, scoring='accuracy')
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    
    precisao, revocacao, f1score, acuracia, acuracia_balanceada = metricasAvaliacao(y_test, predicted, 'SVM', output_filename)
    
    return [precisao, revocacao, f1score, acuracia, acuracia_balanceada]


def randomForest(x_train, x_test, y_train, y_test, cv, output_filename):
    
    print(output_filename)    
    print('randomForest')
    
    params = {'estimator__criterion':['gini', 'entropy'], 
              'estimator__max_depth':[3,4], 
              'estimator__n_estimators':[80,90]}
    
    clf = GridSearchCV(OneVsRestClassifier(RandomForestClassifier(class_weight='balanced')), params, cv=cv,
                       n_jobs=1, verbose=1, scoring='accuracy')
    
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)

    precisao, revocacao, f1score, acuracia, acuracia_balanceada = metricasAvaliacao(y_test, predicted, 'FlorestaRandomica', output_filename)
    
    return [precisao, revocacao, f1score, acuracia, acuracia_balanceada]


def mlp(x_train, x_test, y_train, y_test, cv, output_filename):
    
    print(output_filename)    
    print('mlp')
    
    params = {'estimator__activation':['relu', 'tanh'], 
              'estimator__hidden_layer_sizes':[(20,)], 
              'estimator__solver':['adam']}
    
    clf = GridSearchCV(OneVsRestClassifier(MLPClassifier()), params, cv=cv,
                       n_jobs=1, verbose=1, scoring='accuracy')
    
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    
    precisao, revocacao, f1score, acuracia, acuracia_balanceada = metricasAvaliacao(y_test, predicted, 'PerceptronMulticamadas', output_filename)
    
    return [precisao, revocacao, f1score, acuracia, acuracia_balanceada]
    

def logisticRegression(x_train, x_test, y_train, y_test, cv, output_filename):
    
    print(output_filename)    
    print('logisticRegression')
    
    params = {'estimator__C':[1], 
              'estimator__penalty':['l2']}
    
    clf = GridSearchCV(OneVsRestClassifier(LogisticRegression(class_weight='balanced')), params, cv=cv,
                       n_jobs=1, verbose=1, scoring='accuracy')
    
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    
    precisao, revocacao, f1score, acuracia, acuracia_balanceada = metricasAvaliacao(y_test, predicted, 'RegressaoLogistica', output_filename)
    
    return [precisao, revocacao, f1score, acuracia, acuracia_balanceada]


def classificaoMultiClasse(dataset, output_filename):
    
    if not path.exists('results/'):
        makedirs('results/')

    df = pd.read_csv('../5_reducaoDimensionalidade/datasets/{}_reduced_pca.csv'.format(dataset), sep=';', header=0)
    
    x = df.drop('_class_', axis=1)
    y = df['_class_']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=183212)
    
    cv = KFold(n_splits=10, shuffle=True, random_state=183212)
    
    metrics_svm = svm(x_train, x_test, y_train, y_test, cv, output_filename)
    metrics_rf = randomForest(x_train, x_test, y_train, y_test, cv, output_filename)
    metrics_mlp = mlp(x_train, x_test, y_train, y_test, cv, output_filename)
    metrics_lr = logisticRegression(x_train, x_test, y_train, y_test, cv, output_filename)
    
    metrics = [metrics_svm, metrics_rf, metrics_mlp, metrics_lr]
    
    columns = ['Precisão', 'Recall', 'F1-Score', 'Acurácia', 'Acurácia Balanceada']   
    index = ['SVM', 'Floresta Randômica', 'Perceptron Multicamadas', 'Regressão Logística']
    
    df_metrics = pd.DataFrame(data=metrics, columns=columns, index=index)  
    df_metrics.index.name = "Classificador"
    
    print(df_metrics, end='\n\n')
        
    df_metrics.to_csv('results/clfs_metrics_{}.csv'.format(output_filename), sep=";")
   

if __name__ == '__main__':
    
    datasets = ['tfidf-11', 'tfidf-12', 'bow-11', 'bow-12']
    
    for dataset in datasets:
        classificaoMultiClasse(dataset, "{}class".format(dataset))

    print("Fim")

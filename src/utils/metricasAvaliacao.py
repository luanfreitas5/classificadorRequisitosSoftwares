#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from os import path, makedirs
from warnings import filterwarnings
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
filterwarnings('ignore')

def metricasAvaliacao(y_test, predicted, name_clf, output_filename):
    
    precisao = np.round(precision_score(y_test, predicted, average='weighted') * 100, 2)
    revocacao = np.round(recall_score(y_test, predicted, average='weighted') * 100, 2)
    f1score = np.round(f1_score(y_test, predicted, average='weighted') * 100, 2)
    acuracia = np.round(accuracy_score(y_test, predicted) * 100, 2)
    acuracia_balanceada = np.round(balanced_accuracy_score(y_test, predicted) * 100, 2)
    
    if not path.exists('matrizesConfusoes/{}/'.format(name_clf)):
        makedirs('matrizesConfusoes/{}/'.format(name_clf))
        
    matriz_confusao = confusion_matrix(y_test, predicted)
    matriz_confusao_relativa = matriz_confusao / matriz_confusao.sum(axis=1, keepdims=True)
        
    numero_classes = int(re.findall(r'\w+-(\d+)\w+', output_filename)[0])
    if(numero_classes == 2):
        class_names = ['F', 'NF']
    elif(numero_classes == 11):
        class_names = ['A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']
    else:
        class_names = ['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']
        
    plt.figure(figsize=(10, 8))
    plt.title("Matriz de Confusao do {} da base {}".format(name_clf, output_filename))
    sns.heatmap(matriz_confusao_relativa, linewidths=.5, cmap='coolwarm', annot=True, fmt='.1%', vmin=0, vmax=1, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.savefig('matrizesConfusoes/{}/matrizConfusao{}{}'.format(name_clf, name_clf, output_filename))
    # plt.show()
    plt.close()
    
    return precisao, revocacao, f1score, acuracia, acuracia_balanceada
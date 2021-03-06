#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from os import path, makedirs
from warnings import filterwarnings
filterwarnings('ignore')


def process_dataset(name, stat_significance_level):
    df = pd.read_csv('./datasets/{}.csv'.format(name), sep=';', header=0)

    X = df.drop('_class_', axis=1)
    y = df['_class_']
    selection = SelectKBest(chi2, k='all').fit(X, y)
    features_criterion = selection.pvalues_ < stat_significance_level

    # output the reduced dataset
    features_criterion_select = np.append(arr=features_criterion, values=[True], axis=0)
    output_df = df.loc[:, features_criterion_select]
    output_df.to_csv('./datasets/{}_reduced.csv'.format(name), sep=';', header=True, index=False)

    # return features count and 10 best feature names
    features_scores_map = { X.columns[index]: selection.scores_[index] for index, _ in enumerate(X.columns) }
    top_ten_feature_names = sorted(features_scores_map, key=features_scores_map.get, reverse=True)[:10]
    
    return (len(output_df.columns) - 1, top_ten_feature_names)


if __name__ == '__main__':
    
    if not path.exists('datasets/'):
        makedirs('datasets/')

    datasets = ['tfidf-2', 'tfidf-11', 'tfidf-12', 'bow-2', 'bow-11', 'bow-12']
    stat_significances = [0.325, 0.8, 0.8, 0.075, 0.075, 0.075]
    infos = [ process_dataset(dataset, stat_significances[index]) for index, dataset in enumerate(datasets) ]
    lines = [ '** {} **\nQuantidade de Caracteristicas: {}\nTop 10 Carateristicas: {}\n\n'.format(datasets[index], info[0], ', '.join(info[1])) for index, info in enumerate(infos) ]
    lines[len(lines) - 1] = lines[len(lines) - 1][:-1]  # cut off extraneous newline
    
    with open('./datasets/chiSquared_info.txt', 'w') as writer:
        writer.writelines(lines)
        
    print("Fim da selecao de caracteristicas com teste de hipotese Qui-Quadrado")

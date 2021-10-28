#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.decomposition import PCA
from os import path, makedirs
from warnings import filterwarnings
filterwarnings('ignore')


def transform_dataset(name, component_count):
    df = pd.read_csv('../4_selecaoCaracteristicas/datasets/{}_reduced.csv'.format(name), sep=';', header=0)
    X = df.drop('_class_', axis=1)

    pca = PCA(n_components=component_count)
    pca.fit(X)
    print('{:<9}: {:.3f}%'.format(name, sum(pca.explained_variance_ratio_[:component_count]) * 100))

    X_new = pca.transform(X)
    X_new_cols = [ 'Comp{}'.format(index + 1) for index in range(X_new.shape[1]) ]
    df_output = pd.DataFrame(data=X_new, columns=X_new_cols)
    df_output['_class_'] = df['_class_']

    df_output.to_csv('./datasets/{}_reduced_pca.csv'.format(name), sep=';', header=True, index=False)
    
    return len(df_output.columns) - 1

if __name__ == '__main__':
    
    if not path.exists('datasets/'):
        makedirs('datasets/')
        
    print('** Explained Variance **')
        
    datasets = ['bow-12', 'bow-11', 'bow-2', 'tfidf-12', 'tfidf-11', 'tfidf-2']
    component_counts = [300, 200, 250, 300, 200, 300]  # based on slope of explained variance cumulative sum curves from 3-pca-investigate.py (98% - 99%)

    #for index, dataset in enumerate(datasets):
    #    transform_dataset(dataset, component_counts[index])
        
        
    infos = [ transform_dataset(dataset, component_counts[index]) for index, dataset in enumerate(datasets) ]
    lines = [ '** {} **\nQuantidade de Caracteristicas: {}\n\n'.format(datasets[index], info) for index, info in enumerate(infos) ]
    lines[len(lines) - 1] = lines[len(lines) - 1][:-1]  # cut off extraneous newline
    
    with open('./datasets/pca_info.txt', 'w') as writer:
        writer.writelines(lines)
        
    print("Fim da reducao de dimensionalidade PCA")

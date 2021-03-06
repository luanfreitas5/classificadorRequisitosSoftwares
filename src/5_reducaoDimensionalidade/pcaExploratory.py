#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import ticker
from os import path, makedirs
from warnings import filterwarnings
filterwarnings('ignore')


def investigate_dataset(name, axs):
    df = pd.read_csv('../4_selecaoCaracteristicas/datasets/{}_reduced.csv'.format(name), sep=';', header=0)
    X = df.drop('_class_', axis=1)

    components_count = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=components_count)
    pca.fit(X)

    explained_variance_cumsum = pd.Series(data=pca.explained_variance_ratio_).sort_values(ascending=False).cumsum()
    
    axs.set_title(name)
    axs.plot(explained_variance_cumsum)
    axs.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axs.set_yticks(np.linspace(0, 1, 11), minor=False)
    axs.set_yticks(np.linspace(0, 1, 21), minor=True)
    axs.grid(color='#e0e0e0', which='major')
    axs.grid(color='#f2f2f2', which='minor')


if __name__ == '__main__':
    
    if not path.exists('figures/'):
        makedirs('figures/')
        
    datasets = ['bow-12', 'bow-11', 'bow-2', 'tfidf-12', 'tfidf-11', 'tfidf-2']

    fig, axs = plt.subplots(2, 3)
    fig.suptitle('PCA Explained Variance Cumulative Sums')
    fig.set_size_inches(16, 12)
    
    for index, dataset in enumerate(datasets):
        investigate_dataset(dataset, axs[math.floor(index / 3), index % 3])
    
    fig.savefig('./figures/pcaExploratory.png')
    plt.show()
    plt.close()
    print("Fim da exploracao do PCA")
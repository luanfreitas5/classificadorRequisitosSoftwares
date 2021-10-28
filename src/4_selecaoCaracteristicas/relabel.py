#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from os import path, makedirs
from warnings import filterwarnings
filterwarnings('ignore')

if __name__ == '__main__':
    
    if not path.exists('datasets/'):
        makedirs('datasets/')

    bow_df = pd.read_csv('../3_extracaoCaracteristicas/datasets/dataset_bow.csv', sep=';', header=0)
    tfidf_df = pd.read_csv('../3_extracaoCaracteristicas/datasets/dataset_tfidf.csv', sep=';', header=0)
    
    # 12-class datasets (no changes, rename for consistency)
    bow_twelve_df = bow_df
    bow_twelve_df.to_csv('./datasets/bow-12.csv', sep=';', header=True, index=False)
    
    print("dataset bow-12.csv gerado")
    
    tfidf_twelve_df = tfidf_df
    tfidf_twelve_df.to_csv('./datasets/tfidf-12.csv', sep=';', header=True, index=False)
    
    print("dataset tfidf-12.csv gerado")
    
    # 11-class datasets
    bow_eleven_df = bow_df.copy()
    bow_eleven_df = bow_eleven_df[bow_eleven_df['_class_'] != 'F']
    bow_eleven_df.to_csv('./datasets/bow-11.csv', sep=';', header=True, index=False)
    
    print("dataset bow-11.csv gerado")
    
    tfidf_eleven_df = tfidf_df.copy()
    tfidf_eleven_df = tfidf_eleven_df[tfidf_eleven_df['_class_'] != 'F']
    tfidf_eleven_df.to_csv('./datasets/tfidf-11.csv', sep=';', header=True, index=False)
    
    print("dataset tfidf-11.csv gerado")
    
    # 2-class datasets
    binarize_class = lambda entry: 'F' if entry == 'F' else 'NF'
    
    bow_two_df = bow_df.copy()
    bow_two_df['_class_'] = bow_two_df['_class_'].apply(binarize_class)
    bow_two_df.to_csv('./datasets/bow-2.csv', sep=';', header=True, index=False)
    
    print("dataset bow-2.csv gerado")
    
    tfidf_two_df = tfidf_df.copy()
    tfidf_two_df['_class_'] = tfidf_two_df['_class_'].apply(binarize_class)
    tfidf_two_df.to_csv('./datasets/tfidf-2.csv', sep=';', header=True, index=False)
    
    print("dataset tfidf-2.csv gerado")

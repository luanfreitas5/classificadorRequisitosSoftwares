#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from os import path, makedirs
from warnings import filterwarnings
filterwarnings('ignore')

if __name__ == '__main__':
    
    if not path.exists('datasets/'):
        makedirs('datasets/')

    df = pd.read_csv('../2_normalizacaoTextos/datasets/dataset_normalized.csv', sep=';', header=0, quotechar='"', doublequote=True)
    
    tfidf = TfidfVectorizer()
    tfidf_transform = tfidf.fit_transform(df['RequirementText'])
    tfidf_df = pd.DataFrame(tfidf_transform.toarray(), columns=tfidf.get_feature_names())
    tfidf_df['_class_'] = df['_class_']
    tfidf_df.to_csv('./datasets/dataset_tfidf.csv', sep=';', header=True, index=False)
    
    print("Fim da vetorizacao tfidf")
    
    bow = CountVectorizer()
    bow_transform = bow.fit_transform(df['RequirementText'])
    bow_df = pd.DataFrame(bow_transform.toarray(), columns=bow.get_feature_names())
    bow_df['_class_'] = df['_class_']
    bow_df.to_csv('./datasets/dataset_bow.csv', sep=';', header=True, index=False)
    
    print("Fim da vetorizacao bow (bag of words)")
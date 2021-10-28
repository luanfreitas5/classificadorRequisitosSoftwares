#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from collections import defaultdict
from nltk.corpus import wordnet as wn, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from os import path, makedirs
from warnings import filterwarnings
filterwarnings('ignore')

# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# wordnet
# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

def normalize(text):
    # convert to lowercase and tokenize (a token is either a word or interpunction symbols)
    tokens = word_tokenize(text.lower())

    # filter out non-alphabetic strings, and stopwords ('a', 'the', etc.)
    # what remains will either be nouns, verbs, adjectives or adverbs
    # finally, perform lemmatization, i.e. replacing each word with its "root" form (e.g.: plural to singular form for nouns, past to present tense for verbs, etc.)
    resulting_words = []
    lemmatizer = WordNetLemmatizer()

    english_stopwords = stopwords.words('english')
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for word, tag in pos_tag(tokens):
        if word.isalpha() and word not in english_stopwords:
            resulting_words.append(
                lemmatizer.lemmatize(word, tag_map[tag[0]]))

    return ' '.join(resulting_words)


if __name__ == '__main__':
    
    if not path.exists('datasets/'):
        makedirs('datasets/')
    
    df = pd.read_csv('../1_exploracaoDados/datasets/PROMISE_exp.csv', sep=',', header=0, quotechar='"', doublequote=True)

    del df['ProjectID']

    df['RequirementText'] = df['RequirementText'].apply(normalize)
    
    df.to_csv('./datasets/dataset_normalized.csv', sep=';', header=True, index=False, quotechar='"', doublequote=True)
    
    print("Fim da Normalizacao")
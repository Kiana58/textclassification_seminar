#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Basic Test Preprocessing functions
"""

import pandas as pd
import gensim
from nltk.stem.snowball import EnglishStemmer
stemmer = EnglishStemmer()
import numpy as np
from pprint import pprint
import string
import re

import os
import sys
reload(sys)  
sys.setdefaultencoding('utf8')

def load_data(path):
    # set wd
    os.chdir(path)

    # load data
    train = pd.read_csv("data/train_binary.csv")
    test = pd.read_csv("data/test_binary.csv")

    # convert label into numeric
    train['y'] =  pd.get_dummies(train["Category"])["cancer"]
    test['y'] = None

    # merge train and test set as design matrix
    df = train.append(test)
    
    return train, test, df    

# function to remove stopwords, retrun TRUE if word is not in the stoplist  
def noStopWord(word,stoplist):
    return word not in stoplist


# basic preprocessing before convert into bag-of-words or tf-idf
def basic_preprocesser(df, stoplist, max_freq=0.9, min_freq=5):
    
    documents = df["Title"]

    # check for stopwords and perform stemming
    texts_stem = [[stemmer.stem(word) for word in document.lower().split() 
                    if noStopWord(word,stoplist)] for document in documents]
    
    return texts_stem

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

''' 
CountVectorizer perform all basic text processing steps in one function
'''
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# load data
path = '/home/dkohn/kaggle/textclassification_seminar'
train, test, df = load_data(path)
target = train['y']
documents = train["Abstract"]

# apply preprocessing on text
analyzer = CountVectorizer().build_analyzer()
count_vect = CountVectorizer(binary=False,              
                             analyzer=stemmed_words,    # use predefined function for stemming
                             stop_words='english',      # remove stop words
                             max_df=0.9,                # max fraction a word appears in a document
                             min_df=5,                  # min appearance of a word in documuent
                             lowercase=True, 
                             strip_accents="unicode"
                             )
# get bag-of-words
count_vect.fit(documents)
X_bow = count_vect.transform(documents)

# tf-idf
tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(X_bow)

# for embeddings input
# load stopword list
stoplist = list(gensim.parsing.preprocessing.STOPWORDS)

# use basic_preprocessor
texts = basic_preprocesser(df,stoplist)
dictionary = gensim.corpora.Dictionary(texts)
dictionary.filter_extremes(no_above=0.9, no_below=5)

# actual Bag-of_Words using CountVectorizer
count_vect = CountVectorizer(vocabulary=dictionary.token2id)
documents = df["Title"]
X_bow = count_vect.fit_transform(documents)

# train and test design matrix and labels
X_train_bow = X_bow[0:2155]
y_train = df['y'][0:2155]
X_test_bow = X_bow[2155:2175]
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Script to tune a simple Feed-Forward Net 

@author: dkohn
"""
import pandas as pd
import numpy as np
from scipy import sparse

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from keras.regularizers import l1, l2
from keras.utils import np_utils

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.cross_validation import KFold

import itertools # for grid

# load relevant functions
from helper import score_prediction
from benchmarkNet import simpleMLP
from basic_preprocessing import load_data, stemmed_words

# create a grid for given a list of tuning parameter
def createGrid(tuning_parameters):
    grid = []
    for element in itertools.product(*tuning_parameters):
        grid.append(element)
    return grid


""" Specify Tuning parameters """
# model parameter
nodes_per_layer = [10, 15, 20, 25, 30] 
weight_decay = [0, 0.00001, 0.0001, 0.001] # values > 0.01 destroy the model 
dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
hidden_activation = ["relu", "tanh"] # activation for input to hidden

# training parameter
solver = ['Adagrad', 'Adadelta']  
regulization_method = ['l1','l2'] 
batch_size = 265 
epoch = 50  # mainly computational 

""" Simple Preprocessing """
# load data
path = '/home/dkohn/kaggle/textclassification_seminar'
train, test, df = load_data(path)
#train = shuffle(train) # shuffle the data (recommended but does not work properly)
target = train['y']
documents = train["Abstract"]

# shuffle the data (recommended but does not work properly)
#documents, target = shuffle(documents, target)

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

""" create grid of parameters """
tuning_parameters = [nodes_per_layer, dropout_rate, hidden_activation, weight_decay]
grid = createGrid(tuning_parameters)

max_acc = 0.5  # best auc so far 
best_parameter = {'nodes': 10, 'dropout_rate': 0.25, 'activation': "relu", "weight_decay" : 0}

average_acc = np.array([]) # array to store all average aucs for different parameters
sd_acc = np.array([]) # to store all standard deviation of aucs


""" Start actual tuning """
for i in range(len(grid)):        
    print('---'*30)
    print('remaining iterations:', len(grid) - i)
    print('best auc so far is:', max_acc)
    print('best set of parameters are', best_parameter)
    
    current_parameters = {'nodes': grid[i][0], 'dropout_rate': grid[i][1], 
                       'activation': grid[i][2], "weight_decay": grid[i][3] }
    print('Now try: ', current_parameters)
    
    nb_folds = 10
    kfolds = KFold(len(target), nb_folds)
    #av_roc = 0.
    
    acc = np.array([]) # array to store all aucraccies in each fold
    f = 0
    for train, valid in kfolds:
        
        print('---'*20)
        print('Fold', f+1)
        
        # counting folds
        f += 1
        
        # splitting the folds       
        X_train = X[train]
        X_valid = X[valid]
        Y_train = target[train]
        Y_valid = target[valid]
        y_valid = target[valid] # for auc calculation
        

        print("Training model...")      
        model = simpleMLP(design_matrix=X_train, 
                           nodes_per_layer=grid[i][0], 
                           dropout_rate=grid[i][1],
                           hidden_activation=grid[i][2],
                           weight_decay=grid[i][3],
                           learning_rate=0.1
                           )        

        # fitting the model
        model.fit(X_train.todense(), 
                  Y_train, 
                  epochs=epoch, 
                  batch_size=batch_size,
                  verbose=0
                  )
        
        # prediction  
        yhat = model.predict(X_valid.todense())
        
        acc_iteration = score_prediction(y_valid, yhat, acc_only=True)
        #auc_iteration = metrics.roc_auc_score(y_valid, valid_preds)
        print("Accuraccy:", acc_iteration)
        print('---'*20)
        
        # save auc score of current iteration
        acc = np.append(acc,acc_iteration)
    
    acc_mean = np.mean(acc)
    acc_std = np.std(acc)
    
    if acc_mean > max_acc:
        max_acc = acc_mean
        best_parameter = current_parameters
    
    average_acc = np.append(average_acc, acc_mean) 
    sd_acc = np.append(sd_acc, acc_std)
    
    print('****'*20)
    print('****'*20)
    print('Average AUC:', acc_mean, 'Std:', acc_std)    
    print('Parameters: ', current_parameters)
    print('****'*20)
    print('****'*20)    



model_results =  np.column_stack((average_acc, sd_acc, np.array(grid))) 
print model_results
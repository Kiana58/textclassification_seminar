#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 14:13:28 2017

@author: dkohn

"""
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.layers import Embedding, Conv1D, GlobalMaxPool1D
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, RMSprop
from keras.regularizers import l1, l2

# MLP with single hidden layer and l2 regulization, no dropout
def simpleMLP(design_matrix, nodes_per_layer, dropout_rate, weight_decay,
              hidden_activation="relu", learning_rate=0.1):

    model = Sequential()    # feed-forward model

    # input to hidden layer
    model.add(Dense(input_shape=(design_matrix.shape[1],),   # input dimension
                    units=nodes_per_layer,                   # output dimension
                    activation=hidden_activation             # activation functio
                    )
    )

    if dropout_rate > 0:
        # add dropout
        model.add(Dropout(dropout_rate))

    # hidden to output layer
    model.add(Dense(input_shape=(nodes_per_layer,),           # input dimension must match the previous output_dim
                    units=1,                                  # output dimension must match the target dimension
                    activation='sigmoid',                     # activation function,
                    activity_regularizer=l2(weight_decay)   # weight decay
                    )
    )

    # specify optimizer for backpropagation
    bp = SGD(lr=learning_rate)

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer=bp, class_mode="binary", metrics=['accuracy'])

    return model

def basic_ffnet(input_shape):
    model = Sequential()
    model.add(Dense(16, input_shape=input_shape, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 1-CNN with embedded layer not tested yet, no gpu :(
def embeddingsCNN(dictionary, embedded_dim, learning_rate=0.01,
                  feature_maps=100, window_size=3, dropout_rate=0.25):

    # transform dictionary into suitable input
    token2id = dictionary.token2id


    model = Sequential()    # feed-forward model

    # embedded layer (has no activation function)
    model.add(Embedding(input_dim=len(token2id),       # input dimension
                        output_dim=embedded_dim       # output dimension of embedded layer
                )
             )

    # add dropout (set higher than after convolutional layer)
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(filters=feature_maps,         # feature maps
                     kernel_size=window_size,      # length of the 1D convolution windows (can be a tuple or list?)
                     padding='valid',
                     activation='relu'             # usually tanh or relu
                     )
    )
    model.add(GlobalMaxPool1D())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation ='sigmoid'))

    bp = SGD(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=bp, class_mode="binary", metrics=['accuracy'])

    return model
    
def simpleCNN(output_dim=23, maxlen=100, embedding_length=200, filters = 400, kernel_size = 7, 
              dropout_rate=0.5, nodes_per_layer=128, learning_rate=0.001):
    
    model = Sequential()    # feed-forward model
    
    # convolutional layer
    model.add(Conv1D(filters=filters,
                 kernel_size = kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1, 
                 input_shape=(maxlen, embedding_length)
                 )
    )
    
    model.add(GlobalMaxPool1D())
    model.add(Dropout(dropout_rate))
    
    # intermediate dense layer
    model.add(Dense(nodes_per_layer, activation="relu"))
    model.add(Dropout(dropout_rate))
    
    # output layer
    model.add(Dense(output_dim, activation='sigmoid'))
    
    # optimization
    bp = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=bp, class_mode="binary", metrics=['accuracy'])
    
    return model

"""
This scipt tests the word vectors save in the pubmed script
"""
import numpy as np 
import helper

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, Conv2D, GlobalMaxPool1D,MaxPool1D, GlobalMaxPool2D,Dropout, Dense, Reshape, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

X_train = np.load(dir_name + 'X_train_seeds42.npy')
y_train = np.load(dir_name + 'y_train_seeds42.npy')
X_test = np.load(dir_name + 'X_test_seeds42.npy')
y_test = np.load(dir_name + 'y_test_seeds42.npy')
    

"""
putting data into a CNN 
"""

def create_cnn_model(output_dim=1):
    input_shape = (maxlen, embedding_length)
    filters = 400
    kernel_size = (7)
    model = Sequential()
    model.add(Conv1D(400,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1, input_shape=input_shape))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='sigmoid'))

    return model

model = create_cnn_model(output_dim=23)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=256)
yhat = model.predict(X_test)
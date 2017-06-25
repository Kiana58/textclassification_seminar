"""
This scipt tests the word vectors save in the pubmed script
"""
import numpy as np 
from gc import collect 
from helper import score_prediction 
from benchmarkNet import simpleCNN 
from pubmed import load_prepared_train_data, load_prepared_test_data

file_name = 'data/word_vectors/pubmed_maxlen100_embeddinglen200_'
X_train, y_train = load_prepared_train_data(file_name)
    
"""
putting data into a CNN 
"""
model = simpleCNN() 
model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=256)

# load test data
X_test, y_test = load_prepared_test_data(file_name)

# predict and evaluate
yhat = model.predict(X_test)
score_prediction(y_test, yhat, binary=False)
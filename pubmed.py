import numpy as np

from gc import collect
from helper import load_multiclass_data

from gensim.models.keyedvectors import KeyedVectors
from gensim.parsing.preprocessing import STOPWORDS

from nltk.tokenize import RegexpTokenizer

from sklearn.model_selection import train_test_split

# define tokenizer and stopwords 
tokenizer = RegexpTokenizer(r'\w+')
stoplist = list(STOPWORDS) + ["<span"] + list(range(0,9))

embedding_length = 200
default = np.zeros(embedding_length)


no_hit = ()
def word2vec(word, word_vectors):
    global no_hit
    try:
        return word_vectors.word_vec(word)
    except KeyError as e:
        no_hit = no_hit + (word,)
        return default

def word_is_valid(word):
    return word not in stoplist and len(word) > 2 and len(word) < 20

def pad_or_trunc(doc, maxlen=100):
    if len(doc) < maxlen:
        return np.concatenate((doc, np.zeros((maxlen-len(doc), 200))))
    else:
        return doc[:maxlen]

def load_prepared_train_data(file_name):
    print('Load training data')
    X_train = np.load(file_name + 'X_train.npy')
    y_train = np.load(file_name + 'y_train.npy')
    return X_train, y_train

def load_prepared_test_data(file_name):
    print('Load training data')
    X_train = np.load(file_name + 'X_test.npy')
    y_train = np.load(file_name + 'y_test.npy')
    return X_train, y_train 
    
def prepare_word_vector(input_file, output_file, maxlen=100, embedding_len=200):
    
    print('load train set and word vectors')
    documents_train, target_train = load_multiclass_data()
    word_vectors = KeyedVectors.load_word2vec_format('data/' + input_file, binary=True)
    
    print('match word_vectors with data')
    texts = [np.array([np.transpose(word2vec(word)) for word in 
          tokenizer.tokenize(document.lower()) if word_is_valid(word)]) for document in documents_train]
                       
    del word_vectors, documents_train
    collect()
    
    print('perform padding')
    texts_padded = np.array([pad_or_trunc(text) for text in texts])
    del texts
    collect()
    
    X_train, X_test, y_train, y_test = train_test_split(texts_padded, target_train.values, test_size=0.1, random_state=42)
    
    file_name = output_file + '_malen_' * str(maxlen) + '_embeddinglen_' + str(embedding_len) 
    np.save(file_name + 'X_train.npy', X_train)
    np.save(file_name + 'y_train.npy', y_train)
    np.save(file_name + 'X_test.npy', X_test)
    np.save(file_name + 'y_test.npy', y_test)



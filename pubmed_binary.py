import numpy as np
import gc
from helper import load_multiclass_data, load_binary_data

from gensim.models.keyedvectors import KeyedVectors
from gensim.parsing.preprocessing import STOPWORDS

from nltk.tokenize import RegexpTokenizer

from sklearn.model_selection import train_test_split


# load train data and pretrained word vectors
documents_train, target_train = load_binary_data()
word_vectors = KeyedVectors.load_word2vec_format('data/PubMed-and-PMC-w2v.bin', binary=True)

# define tokenizer and stopwords
tokenizer = RegexpTokenizer(r'\w+')
stoplist = list(STOPWORDS) + ["<span"] + list(range(0, 9))

embedding_length = 200
default = np.zeros(embedding_length)

no_hit = ()
def word2vec(word):
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

texts = [np.array([np.transpose(word2vec(word)) for word in
          tokenizer.tokenize(document.lower()) if word_is_valid(word)]) for document in documents_train]

del word_vectors, documents_train
gc.collect()

texts_padded = np.array([pad_or_trunc(text) for text in texts])
del texts
gc.collect()


"""
save the train set
"""

X_train, X_test, y_train, y_test = train_test_split(texts_padded, target_train.values,
                                                    test_size=0.1, random_state=42)

dir_name = 'data/word_vectors/pubmed_binary_stratify_no_wiki_'
np.save(dir_name + 'X_train_seeds42.npy', X_train)
np.save(dir_name + 'y_train_seeds42.npy', y_train)
np.save(dir_name + 'X_test_seeds42.npy', X_test)
np.save(dir_name + 'y_test_seeds42.npy', y_test)

del X_train, X_test, y_train, y_test
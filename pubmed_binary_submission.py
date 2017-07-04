import pandas as pd
from gensim.models import KeyedVectors
import numpy as np
from nltk import RegexpTokenizer
from gensim.parsing.preprocessing import STOPWORDS

df = pd.read_csv("data/submission_binary.csv")
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
          tokenizer.tokenize(document.lower()) if word_is_valid(word)]) for document in df["Abstract"].values]

print(f"Number of unmatched words: {len(no_hit)}")

texts_padded = np.array([pad_or_trunc(text) for text in texts])


dir_name = 'data/submission_binary_'
np.save(dir_name + 'word_vectors.npy', texts_padded)
np.save(dir_name + 'ids.npy', df["Id"])
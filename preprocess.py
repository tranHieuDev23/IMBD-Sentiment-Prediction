from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np

nltk.download('punkt')

ps = PorterStemmer()


def preprocess_input(text):
    text = text.strip().lower()
    tokens = word_tokenize(text)
    stemmed = [ps.stem(item) for item in tokens]
    return stemmed


def get_token_to_id_dict(X):
    vocab = set()
    for item in X:
        for token in item:
            vocab.add(token)
    token_to_id = {}
    id = 0
    for token in vocab:
        id += 1
        token_to_id[token] = id
    return token_to_id


def get_id_vector(tokens, token_to_id):
    result = []
    for token in tokens:
        if (token in token_to_id):
            result.append(token_to_id[token])
        else:
            result.append(0)
    return result

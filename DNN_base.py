"""
Base model for the DNN used taken from

https://github.com/bundinyo/lime/blob/5b1f754a4e266a451ed9bf63525bc64e7bd567f7/lstm_lime.py
"""

import numpy as np
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator, TransformerMixin


class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):
    """
    Sklearn transformer to convert texts to indices list
        (e.g. [["the cute cat"], ["the dog"]] -> [[1, 2, 3], [1, 4]])
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, texts, y=None):
        self.fit_on_texts(texts)
        return self

    def transform(self, texts, y=None):
        return np.array(self.texts_to_sequences(texts))


class Padder(BaseEstimator, TransformerMixin):
    """
    Pad and crop uneven lists to the same length.
    Only the end of lists longernthan the maxlen attribute are
    kept, and lists shorter than maxlen are left-padded with zeros
    Attributes
    ----------
    maxlen: int
        sizes of sequences after padding
    max_index: int
        maximum index known by the Padder, if a higher index is met during
        transform it is transformed to a 0
    """

    def __init__(self, maxlen=500):
        self.maxlen = maxlen
        self.max_index = None

    def fit(self, X, y=None):
        self.max_index = pad_sequences(X, maxlen=self.maxlen).max()
        return self

    def transform(self, X, y=None):
        X = pad_sequences(X, maxlen=self.maxlen)
        X[X > self.max_index] = 0
        return X


def create_model():
    model = Sequential()
    model.add(Embedding(20000, 64, input_length=140, trainable=True))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(LSTM(100))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

import os
import os.path
import sys

import numpy as np
import pandas as pd
import sklearn.linear_model
import spacy
from anchor import anchor_text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

sys.path.insert(0, '..')

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def load_dataset(dataset):
    if dataset == "polarity":
        df = pd.read_csv('../data/' + dataset_name + '_tweets.csv', encoding='utf-8')
        X = df['tweet'].values
        y = df['class'].values

    elif dataset == "hate":
        df = pd.read_csv('../data/' + dataset_name + '_tweets.csv', encoding='utf-8')
        # Removing the offensive comments, keeping only neutral and hatespeech,
        # and convert the class value from 2 to 1 for simplification purposes
        df = df[df['class'] != 1]
        X = df['tweet'].values
        y = df['class'].apply(lambda x: 1 if x == 2 else 0).values

    elif dataset == "liar":
        df_train = pd.read_csv(os.path.join(__location__, "../data/liar_dataset/train.tsv"), encoding='utf-8', sep='\t')
        df_test = pd.read_csv(os.path.join(__location__, "../data/liar_dataset/test.tsv"), encoding='utf-8', sep='\t')
        df_val = pd.read_csv(os.path.join(__location__, "../data/liar_dataset/valid.tsv"), encoding='utf-8', sep='\t')

        mapping = {'pants-fire': 0,
                   'false': 2,
                   'barely-true': 2,
                   'half-true': 2,
                   'mostly-true': 2,
                   'true': 1}

        df_train.iloc[:, 1] = df_train.iloc[:, 1].apply(lambda x: mapping[x])
        df_test.iloc[:, 1] = df_test.iloc[:, 1].apply(lambda x: mapping[x])
        df_val.iloc[:, 1] = df_val.iloc[:, 1].apply(lambda x: mapping[x])

        # Removing middle columns
        df_train = df_train[df_train.iloc[:, 1] != 2]
        df_test = df_test[df_test.iloc[:, 1] != 2]
        df_val = df_val[df_val.iloc[:, 1] != 2]

        X_train = df_train.iloc[:, 2].values
        y_train = df_train.iloc[:, 1].values
        X_test = df_test.iloc[:, 2].values
        y_test = df_test.iloc[:, 1].values
        X_val = df_val.iloc[:, 2].values
        y_val = df_val.iloc[:, 1].values

        Xtt = np.append(X_train, X_test)
        ytt = np.append(y_train, y_test)
        X = np.append(Xtt, X_val)
        y = np.append(ytt, y_val)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.25)

    return X_test, y_test


dataset_name = 'hate'
data, labels = load_dataset(dataset_name)
train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(data, labels, test_size=.2,
                                                                                  random_state=42)
train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1,
                                                                                random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)

print(len(train))
print(len(test))
print(len(val))

vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)
train_vectors = vectorizer.transform(train)
test_vectors = vectorizer.transform(test)
val_vectors = vectorizer.transform(val)

c = sklearn.linear_model.LogisticRegression()
# c = sklearn.ensemble.RandomForestClassifier(n_estimators=500, n_jobs=10)
c.fit(train_vectors, train_labels)
preds = c.predict(val_vectors)
print('Val accuracy', sklearn.metrics.accuracy_score(val_labels, preds))


def predict_lr(texts):
    return c.predict(vectorizer.transform(texts))


nlp = spacy.load('en_core_web_sm')
explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=True)

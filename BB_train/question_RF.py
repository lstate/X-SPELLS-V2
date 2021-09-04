"""
Train a RF black box model for the questions dataset.

Also calculate fidelity of LIME explanations when using the RF used for the fidelity experiment
"""

import csv
import pickle
import sys
import warnings
from statistics import stdev

import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

sys.path.insert(0, '..')
from preprocessing.pre_processing import preProcessing
from lime.lime_text import LimeTextExplainer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def calculate_fidelity():
    # Lime explainers assume that classifiers act on raw text, but sklearn classifiers act on
    # vectorized representation of texts (tf-idf in this case). For this purpose, we will use
    # sklearn's pipeline, and thus implement predict_proba on raw_text lists.
    c = make_pipeline(vectorizer, loaded_model)
    print(c.predict_proba)

    # Creating an explainer object. We pass the class_names as an argument for prettier display.
    explainer = LimeTextExplainer(class_names=class_names)

    ids = list()
    fidelities = list()

    # for i in range(len(X_test)):
    for i in range(100):
        print('index', i)
        # Generate an explanation with at most n features for a random document in the test set.
        idx = i
        exp = explainer.explain_instance(X_test[idx], c.predict_proba, num_features=10)

        label = loaded_model.predict(test_vectors[idx])[0]
        # label = label // 2
        print(label)
        bb_probs = explainer.Zl[:, label]
        bb_probs = np.clip(bb_probs, 0, 1)
        print('bb_probs: ', bb_probs)
        lr_probs = explainer.lr.predict(explainer.Zlr)
        lr_probs = np.clip(lr_probs, 0, 1)
        print('lr_probs: ', lr_probs)
        fidelity = np.sum(np.abs(bb_probs - lr_probs) < 0.05) / len(bb_probs)
        print('fidelity: ', fidelity)
        ids.append(i)
        fidelities.append(fidelity)

    fidelity_average = 0

    for i in range(len(ids)):
        print(ids[i])
        print(fidelities[i])
        fidelity_average += fidelities[i]

    print("fidelity average is: ", fidelity_average / len(ids))
    print("fidelity stdev is:", stdev(fidelities))

    with open('../output/LIME_q_RF.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(ids)):
            writer.writerow([ids[i], 'hate speech', 'RF', fidelities[i]])


df_train = pd.read_csv("../data/question_dataset/question_train.txt", encoding='ISO-8859-1', sep=':',
                       error_bad_lines=False, header=None)
df_test = pd.read_csv("../data/question_dataset/question_test.txt", encoding='ISO-8859-1', sep=':',
                      error_bad_lines=False, header=None)


def remove_first_word(string):
    return string.partition(' ')[2]


df_train.iloc[:, 1] = df_train.iloc[:, 1].apply(remove_first_word)
df_test.iloc[:, 1] = df_test.iloc[:, 1].apply(remove_first_word)

X_train = df_train.iloc[:, 1].values
y_train = df_train.iloc[:, 0].values
X_test = df_test.iloc[:, 1].values
y_test = df_test.iloc[:, 0].values

X_train = preProcessing(X_train)
X_test = preProcessing(X_test)

X = np.append(X_train, X_test)
y = np.append(y_train, y_test)

(unique, counts) = np.unique(y, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)
print(len(y))
print(len(X))

# Which class to define as 0 depends on the distribution of data.
# We pick the class with the largest number of instances.
mapping = {'DESC': 1,
           'ENTY': 0,
           'ABBR': 1,
           'HUM': 1,
           'NUM': 1,
           'LOC': 1}

df_train.iloc[:, 0] = df_train.iloc[:, 0].apply(lambda x: mapping[x])
df_test.iloc[:, 0] = df_test.iloc[:, 0].apply(lambda x: mapping[x])

X_train = df_train.iloc[:, 1].values
y_train = df_train.iloc[:, 0].values
X_test = df_test.iloc[:, 1].values
y_test = df_test.iloc[:, 0].values

X_train = preProcessing(X_train)
X_test = preProcessing(X_test)

X = np.append(X_train, X_test)
y = np.append(y_train, y_test)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.25)

print(len(X_train))
print(len(X_test))

wordcounts = list()
for sentence in X:
    # with nltk tokenize
    nltk_tokens = nltk.word_tokenize(sentence)
    # naive way, splitting words by spaces
    naive_words = sentence.split(' ')
    wordcounts.append(len(nltk_tokens))

average_wordcount = sum(wordcounts) / len(wordcounts)
no_tweets = len(wordcounts)

print(average_wordcount)
print(no_tweets)

class_names = ['Entity', 'All other classes']

# We'll use the TF-IDF vectorizer, commonly used for text.
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
pickle.dump(vectorizer, open("../models/question_tfidf_vectorizer.pickle", "wb"))

# if we run only fidelity, we need to reload the vectorizer
vectorizer = pickle.load(open("../models/question_tfidf_vectorizer.pickle", 'rb'))

test_vectors = vectorizer.transform(X_test)

# Using random forest for classification.
rf = RandomForestClassifier(class_weight="balanced")
rf.fit(train_vectors, y_train)

# save the model to disk
filename = '../models/question_saved_RF_model.sav'
pickle.dump(rf, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Computing interesting metrics/classification report
pred = loaded_model.predict(test_vectors)
print(classification_report(y_test, pred))
print("The accuracy score is {:.2%}".format(accuracy_score(y_test, pred)))

# Following is used to calculate fidelity for all instances using LIME
# calculate_fidelity()

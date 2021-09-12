"""
Train a RF black box model for the liar dataset.

Also calculate fidelity of LIME explanations when using the RF used for the fidelity experiment
"""
import csv
import pickle
import sys
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

    with open('output/LIME_hs_RF.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(ids)):
            writer.writerow([ids[i], 'hate speech', 'RF', fidelities[i]])


df_biden_train = pd.read_csv("../data/stance/biden_stance_train_public.csv", encoding='utf-8')
df_biden_test = pd.read_csv("../data/stance/biden_stance_test_public.csv", encoding='utf-8')
df_trump_train = pd.read_csv("../data/stance/trump_stance_train_public.csv", encoding='utf-8')
df_trump_test = pd.read_csv("../data/stance/trump_stance_test_public.csv", encoding='utf-8')

mapping = {'AGAINST': 0,
           'NONE': 2,
           'FAVOR': 1}

df_biden_train['label'] = df_biden_train['label'].apply(lambda x: mapping[x])
df_biden_test['label'] = df_biden_test['label'].apply(lambda x: mapping[x])
df_trump_train['label'] = df_trump_train['label'].apply(lambda x: mapping[x])
df_trump_test['label'] = df_trump_test['label'].apply(lambda x: mapping[x])

# Removing middle columns
df_biden_train = df_biden_train[df_biden_train['label'] != 2]
df_biden_test = df_biden_test[df_biden_test['label'] != 2]
df_trump_train = df_trump_train[df_trump_train['label'] != 2]
df_trump_test = df_trump_test[df_trump_test['label'] != 2]

X_train_biden = df_biden_train['text'].values
y_train_biden = df_biden_train['label'].values
X_test_biden = df_biden_test['text'].values
y_test_biden = df_biden_test['label'].values

X_train_trump = df_trump_train['text'].values
y_train_trump = df_trump_train['label'].values
X_test_trump = df_trump_test['text'].values
y_test_trump = df_trump_test['label'].values

X_train_biden = preProcessing(X_train_biden)
X_test_biden = preProcessing(X_test_biden)

X_train_trump = preProcessing(X_train_trump)
X_test_trump = preProcessing(X_test_trump)

X_biden = np.append(X_train_biden, X_test_biden)
y_biden = np.append(y_train_biden, y_test_biden)

X_trump = np.append(X_train_trump, X_test_trump)
y_trump = np.append(y_train_trump, y_test_trump)

"""

(unique, counts) = np.unique(y_biden, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)
print(len(y_biden))
print(len(X_biden))

(unique, counts) = np.unique(y_trump, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)
print(len(y_trump))
print(len(X_trump))

"""

X_total = np.append(X_biden, X_trump)
print(len(X_total))

Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_biden, y_biden, random_state=42, stratify=y_biden,
                                                        test_size=0.25)
Xt_train, Xt_test, yt_train, yt_test = train_test_split(X_trump, y_trump, random_state=42, stratify=y_trump,
                                                        test_size=0.25)

print(len(Xb_train))
print(len(Xb_test))

print(len(Xt_train))
print(len(Xt_test))

wordcounts = list()
for sentence in X_total:
    # with nltk tokenize
    nltk_tokens = nltk.word_tokenize(sentence)
    # naive way, splitting words by spaces
    naive_words = sentence.split(' ')
    wordcounts.append(len(nltk_tokens))

average_wordcount = sum(wordcounts) / len(wordcounts)
no_tweets = len(wordcounts)

print(average_wordcount)
print(no_tweets)

Xb_train = preProcessing(Xb_train)
Xt_train = preProcessing(Xt_train)
Xb_test = preProcessing(Xb_test)
Xt_test = preProcessing(Xt_test)

class_names = ['against', 'favor']

# We'll use the TF-IDF vectorizer, commonly used for text.
vectorizer_b = TfidfVectorizer()
vectorizer_t = TfidfVectorizer()

train_vectors_b = vectorizer_b.fit_transform(Xb_train)
pickle.dump(vectorizer_b, open("../models/stance_biden_tfidf_vectorizer.pickle", "wb"))

train_vectors_t = vectorizer_t.fit_transform(Xt_train)
pickle.dump(vectorizer_t, open("../models/stance_trump_tfidf_vectorizer.pickle", "wb"))

# if we run only fidelity, we need to reload the vectorizer
vectorizer_b = pickle.load(open("../models/stance_biden_tfidf_vectorizer.pickle", 'rb'))
vectorizer_t = pickle.load(open("../models/stance_trump_tfidf_vectorizer.pickle", 'rb'))

test_vectors_b = vectorizer_b.transform(Xb_test)
test_vectors_t = vectorizer_t.transform(Xt_test)

# Using random forest for classification.
rf_b = RandomForestClassifier(class_weight="balanced")
rf_t = RandomForestClassifier(class_weight="balanced")

rf_b.fit(train_vectors_b, yb_train)
rf_t.fit(train_vectors_t, yt_train)

# save the model to disk
pickle.dump(rf_b, open('../models/stance_biden_saved_RF_model.sav', 'wb'))
pickle.dump(rf_t, open('../models/stance_trump_saved_RF_model.sav', 'wb'))

# load the model from disk
loaded_model_b = pickle.load(open('../models/stance_biden_saved_RF_model.sav', 'rb'))
loaded_model_t = pickle.load(open('../models/stance_trump_saved_RF_model.sav', 'rb'))

# Computing interesting metrics/classification report
pred_b = loaded_model_b.predict(test_vectors_b)
print("================ Biden model =================")
print(classification_report(yb_test, pred_b))
print("The accuracy score is {:.2%}".format(accuracy_score(yb_test, pred_b)))

pred_t = loaded_model_t.predict(test_vectors_t)
print("================ Trump model =================")
print(classification_report(yt_test, pred_t))
print("The accuracy score is {:.2%}".format(accuracy_score(yt_test, pred_t)))

# Following is used to calculate fidelity for all instances using LIME
# calculate_fidelity()

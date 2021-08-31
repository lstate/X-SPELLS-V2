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


df_train = pd.read_csv("../data/liar_dataset/train.tsv", encoding='utf-8', sep='\t')
df_test = pd.read_csv("../data/liar_dataset/test.tsv", encoding='utf-8', sep='\t')
df_val = pd.read_csv("../data/liar_dataset/valid.tsv", encoding='utf-8', sep='\t')

mapping = {'pants-fire': 0,
           'false': 0,
           'barely-true': 0,
           'half-true': 1,
           'mostly-true': 1,
           'true': 1}

df_train.iloc[:, 1] = df_train.iloc[:, 1].apply(lambda x: mapping[x])
df_test.iloc[:, 1] = df_test.iloc[:, 1].apply(lambda x: mapping[x])
df_val.iloc[:, 1] = df_val.iloc[:, 1].apply(lambda x: mapping[x])

# Removing middle columns
# df_train = df_train[df_train.iloc[:, 1] != 2]
# df_test = df_test[df_test.iloc[:, 1] != 2]
# df_val = df_val[df_val.iloc[:, 1] != 2]

X_train = df_train.iloc[:, 2].values
y_train = df_train.iloc[:, 1].values
X_test = df_test.iloc[:, 2].values
y_test = df_test.iloc[:, 1].values
X_val = df_val.iloc[:, 2].values
y_val = df_val.iloc[:, 1].values

X_train = preProcessing(X_train)
X_test = preProcessing(X_test)
X_val = preProcessing(X_val)

Xtt = np.append(X_train, X_test)
ytt = np.append(y_train, y_test)
X = np.append(Xtt, X_val)
y = np.append(ytt, y_val)

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

class_names = ['fake-news', 'real-news']

# We'll use the TF-IDF vectorizer, commonly used for text.
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
pickle.dump(vectorizer, open("../models/liar_tfidf_vectorizer.pickle", "wb"))

# if we run only fidelity, we need to reload the vectorizer
vectorizer = pickle.load(open("../models/liar_tfidf_vectorizer.pickle", 'rb'))

test_vectors = vectorizer.transform(X_test)

# Using random forest for classification.
rf = RandomForestClassifier(class_weight="balanced")

'''
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                               n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
'''

rf.fit(train_vectors, y_train)

# save the model to disk
filename = '../models/liar_saved_RF_model.sav'
pickle.dump(rf, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Computing interesting metrics/classification report
pred = loaded_model.predict(test_vectors)
print(classification_report(y_test, pred))
print("The accuracy score is {:.2%}".format(accuracy_score(y_test, pred)))

# Following is used to calculate fidelity for all instances using LIME
calculate_fidelity()

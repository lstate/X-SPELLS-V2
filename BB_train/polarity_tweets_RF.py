"""
Train a RF black box model for the polarity dataset.

Also calculate fidelity of LIME explanations when using the RF used for the fidelity experiment
"""

import csv
import pickle
import re
import string
import sys

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, '..')
from lime.lime_text import LimeTextExplainer
from statistics import stdev


def cleanText(var):
    # replace punctuation with spaces
    var = re.sub('[{}]'.format(string.punctuation), " ", var)
    # remove double spaces
    var = re.sub(r'\s+', " ", var)
    # put in lower case
    var = var.lower().split()
    # remove words that are smaller than 2 characters
    var = [w for w in var if len(w) >= 3]
    # remove stop-words
    # var = [w for w in var if w not in stopwords.words('english')]
    # stemming
    # stemmer = nltk.PorterStemmer()
    # var = [stemmer.stem(w) for w in var]
    var = " ".join(var)
    return var


def preProcessing(pX):
    clean_tweet_texts = []
    for t in pX:
        clean_tweet_texts.append(cleanText(t))
    return clean_tweet_texts


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

    #for i in range(len(X_test)):
    for i in range(100):
        print('index', i)
        # Generate an explanation with at most n features for a random document in the test set.
        idx = i
        exp = explainer.explain_instance(X_test[idx], c.predict_proba, num_features=10)

        label = loaded_model.predict(test_vectors[idx])[0]
        # label = label // 1
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

    with open('output/LIME_po_RF.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(ids)):
            writer.writerow([ids[i], 'polarity', 'RF', fidelities[i]])

_, _, y_train, y_test, X_train, X_test = get_text_data("data/polarity_tweets.csv", "polarity")
class_names = ['negative', 'positive']

# We'll use the TF-IDF vectorizer, commonly used for text.
#vectorizer = TfidfVectorizer(sublinear_tf='false')
#train_vectors = vectorizer.fit_transform(X_train)
#pickle.dump(vectorizer, open("models/polarity_tfidf_vectorizer.pickle", "wb"))

vectorizer = pickle.load(open("models/polarity_tfidf_vectorizer.pickle", 'rb'))

test_vectors = vectorizer.transform(X_test)

# Using random forest for classification.
#rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                            max_depth=1000, max_features=1000, max_leaf_nodes=None,
#                            min_impurity_decrease=0.0, min_impurity_split=None,
#                            min_samples_leaf=4, min_samples_split=10,
#                            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=None,
#                            oob_score=False, random_state=None, verbose=0,
#                            warm_start=False)

#rf.fit(train_vectors, y_train)

# Save the model to disk
filename = 'models/polarity_saved_RF_model.sav'
#pickle.dump(rf, open(filename, 'wb'))

# Load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Computing interesting metrics/classification report
pred = loaded_model.predict(test_vectors)
print(classification_report(y_test, pred))
print("The accuracy score is {:.2%}".format(accuracy_score(y_test, pred)))

# Following is used to calculate fidelity for all instances using LIME
calculate_fidelity()

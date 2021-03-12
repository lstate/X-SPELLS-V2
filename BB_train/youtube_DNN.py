"""
Train a DNN black box model for the hate speech dataset.

Also calculate fidelity of LIME explanations when using the DNN used for the fidelity experiment
"""

import csv
import pickle

import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline

from DNN_base import TextsToSequences, Padder, create_model
from pre_processing import YOUTUBE_get_text_data
from statistics import stdev



def calculate_fidelity():
    # Creating an explainer object. We pass the class_names as an argument for prettier display.
    explainer = LimeTextExplainer(class_names=class_names)

    ids = list()
    fidelities = list()

    for i, e in enumerate(X_test):
        print(str(i + 1) + '.', e)

    #for i in range(len(X_test)):
    for i in range(100):

        print('index: ', i)
        # Generate an explanation with at most n features for a random document in the test set.
        idx = i
        exp = explainer.explain_instance(X_test[idx], loaded_model.predict_proba, num_features=10)
        label = pred[i]
        # label = label//1

        bb_probs = explainer.Zl[:, label].flatten()
        bb_probs = np.clip(bb_probs, 0, 1)
        print('bb_probs: ', bb_probs)
        lr_probs = explainer.lr.predict(explainer.Zlr)
        lr_probs = np.clip(lr_probs, 0, 1)
        print('lr_probs: ', lr_probs)
        fidelity = np.sum(np.abs(bb_probs - lr_probs) < 0.05) / len(bb_probs)
        print('fidelity: ', fidelity)
        #print('np.sum: ', np.sum(np.abs(bb_probs - lr_probs) < 0.01))
        ids.append(i)
        fidelities.append(fidelity)
        print('')

    fidelity_average = 0

    for i in range(len(ids)):
        print(ids[i])
        print(fidelities[i])
        fidelity_average += fidelities[i]

    print("fidelity average is: ", fidelity_average/len(ids))
    print("fidelity stdev is:", stdev(fidelities))

    with open('output/LIME_youtube_DNN_redone.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(ids)):
            writer.writerow([ids[i], 'youtube', 'DNN', fidelities[i]])


_, _, y_train, y_test, X_train, X_test = YOUTUBE_get_text_data("data/YouTube-Spam-Collection-v1/youtube.csv", "youtube")
class_names = ['no spam', 'spam']

sequencer = TextsToSequences(num_words=35000)
padder = Padder(140)
myModel = KerasClassifier(build_fn=create_model, epochs=100)

pipeline = make_pipeline(sequencer, padder, myModel)
pipeline.fit(X_train, y_train)

# Save the model to disk
filename = 'models/youtube_saved_DNN_model_redone.sav'
pickle.dump(pipeline, open(filename, 'wb'))

# Load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Computing interesting metrics/classification report
#pred = pipeline.predict(X_test)
pred = loaded_model.predict(X_test)
print(classification_report(y_test, pred))
print("The accuracy score is {:.2%}".format(accuracy_score(y_test, pred)))

# Following is used to calculate fidelity for all instances using LIME
calculate_fidelity()

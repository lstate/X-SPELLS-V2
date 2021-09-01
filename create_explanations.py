import csv
import heapq
import pickle
from collections import Counter
from functools import partial
from statistics import stdev

import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from nltk.corpus import stopwords
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import decision_tree
from DNN_base import TextsToSequences, Padder, create_model
from VAE_train.train_BVAE import get_text_data
from VAE_train.youtube_train_vae import YOUTUBE_get_text_data
from lstm_vae import inference

sequencer = TextsToSequences(num_words=35000)
padder = Padder(140)
myModel = KerasClassifier(build_fn=create_model, epochs=100)

# below functions are needed for diversity optimization

""" argmax function """


def argmax(values, f):
    bestf = None
    bestp = None
    for i, v in enumerate(values):
        fv = f(v)
        if bestf is None or bestf < fv:
            bestf = fv
            bestp = i
    return bestp, bestf


# euclidian
def dist(a, b):
    return distance.euclidean(a, b)


""" accelerated max cover """


# return argmax_{x subseteq aset} f(x) using accelerated greedy algorithm
def acover(aset, f=None, k=None, df=None, verbose=True):
    goal_function = []
    if k is None or k >= len(aset):
        k = len(aset)
    # priority queue
    if df is None:
        pset = [Pair(-f([v]), v) for v in aset]
    else:
        pset = [Pair(-df([], v), v) for v in aset]
    heapq.heapify(pset)
    # covers is accumulated greedy set
    covers = []
    for i in range(k):
        # get next value
        best = heapq.heappop(pset)
        best.a = -f(covers + [best.b]) if df is None else -df(covers, best.b)
        # and check if it is better than any bound
        newbest = heapq.heappushpop(pset, best)
        while newbest.a < best.a:
            best = newbest
            best.a = -f(covers + [best.b]) if df is None else -df(covers, best.b)
            newbest = heapq.heappushpop(pset, best)
        # add it to the accumulated greedy set
        if df is not None and best.a >= 0:
            break
        covers.append(best.b)
        if verbose:
            if df is None:
                print('it:', i, 'x:', best.b, 'f:', -best.a)
            else:
                print('it:', i, 'x:', best.b, 'f:', df(covers), 'df:', -best.a)
        # safe the values of the goal function
        # df needs to be not None
        goal_function.append(df(covers))
    return covers, goal_function


""" pairs of objects """


class Pair:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    """ lexicographic ordering """

    def __lt__(self, other):
        return self.a < other.a or (self.a == other.a and self.b < other.b)

    """ lexicographic ordering """

    def __le__(self, other):
        return self.a < other.a or (self.a == other.a and self.b <= other.b)

    """ lexicographic ordering """

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b


def load_VAE(dataset_name):
    vae = load_model('models/' + dataset_name + '_vae_model.h5', compile=False)
    enc = load_model('models/' + dataset_name + '_enc_model.h5', compile=False)
    gen = load_model('models/' + dataset_name + '_gen_model.h5', compile=False)
    stepper = load_model('models/' + dataset_name + '_stepper_model.h5', compile=False)
    vae.summary()
    return vae, enc, gen, stepper


def decode(s):
    return inference.decode_sequence(s, gen, stepper, input_dim, char2id, id2char, max_encoder_seq_length)


def get_sentences():
    input_sentences = []
    state_input_sentences = []
    decoded_sentences = []

    for i in range(len(encoder_input_data)):
        mean, variance = enc.predict([[encoder_input_data[i]]])
        seq = np.random.normal(size=(latent_dim,))
        seq = mean + variance * seq
        input_sentences.append(X_original_processed[i])
        state_input_sentences.append(seq)
        decoded_sentences.append(decode(seq))

    return input_sentences, state_input_sentences, decoded_sentences
    # return input_sentences, state_input_sentences


def calculate_MRE():
    train_input_sentences = []
    train_decoded_sentences = []

    for i in range(int(len(encoder_input_data))):
        print(i)
        mean, variance = enc.predict([[encoder_input_data[i]]])
        seq = np.random.normal(size=(latent_dim,))
        seq = mean + variance * seq
        seq = mean
        print('original: ', X_original_processed[i])
        print('reconstructed: ', decode(seq))
        train_input_sentences.append(X_original_processed[i])
        train_decoded_sentences.append(decode(seq))

    train_sentences_dict = dict(zip(train_input_sentences, train_decoded_sentences))
    train_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    train_sentences_vectors = train_vectorizer.fit_transform(train_input_sentences).toarray()
    train_decoded_sentences_vectors = train_vectorizer.transform(train_decoded_sentences).toarray()

    train_cosine_distance_list = list()

    for i in range(len(train_sentences_vectors)):
        train_cosine_distance_list.append((cdist(train_sentences_vectors[i].reshape(1, -1),
                                                 train_decoded_sentences_vectors[i].reshape(1, -1),
                                                 metric='cosine').ravel())[0])

    # print(train_sentences_dict)
    # print(train_cosine_distance_list)
    print("MRE train: ", sum(train_cosine_distance_list) / len(train_cosine_distance_list))
    print("MRE train stdev: ", stdev(train_cosine_distance_list))


def calculate_min_max(list):
    c = np.min(list, axis=0)
    d = np.max(list, axis=0)
    return c, d


def generate_sentences(number_of_sentences, number_of_max_attempts, number_of_random_sentences, probability):
    state_sentences = [[] for _ in range(number_of_sentences)]
    decoded_sentences = [[] for _ in range(number_of_sentences)]

    for i in range(number_of_sentences):

        print("sentence : %d/%d" % (i, number_of_sentences))
        seq_from = latent_space_state[i]

        number_of_ticks = 0
        max_attempts = number_of_max_attempts
        random_sentences_to_create = number_of_random_sentences

        while (len(decoded_sentences[i]) < random_sentences_to_create) and number_of_ticks < max_attempts:

            newseq = np.copy(seq_from)
            for d in range(latent_dim):
                rm = np.random.random()
                if rm >= probability:
                    newseq[0, d] = (largest_x[0, d] - smallest_x[0, d]) * np.random.random() + smallest_x[0, d]

            if decode(newseq) not in decoded_sentences[i]:
                state_sentences[i].append(newseq)
                decoded_sentences[i].append(decode(newseq))

            number_of_ticks += 1
            print('unique/tick ratio: %d/%d' % (len(decoded_sentences[i]), number_of_ticks))

    return state_sentences, decoded_sentences


def get_predictions(bb_filename, vect_filename, number_of_sentences):
    # Load the black box model from disk
    print(bb_filename)
    loaded_model = pickle.load(open(bb_filename, 'rb'))

    preds = [[] for _ in range(number_of_sentences)]

    print('predictions are: ')

    final_unique_state_sentences = [[] for _ in range(number_of_sentences)]  # with initial sentence on first place
    final_unique_decoded_sentences = [[] for _ in range(number_of_sentences)]

    if vect_filename is None:
        for i in range(number_of_sentences):
            final_unique_state_sentences[i].append(latent_space_state[i])
            final_unique_decoded_sentences[i].append(decoded_sentences[i])

            final_unique_state_sentences[i].extend(generated_state_sentences[i])
            final_unique_decoded_sentences[i].extend(generated_decoded_sentences[i])

            # Predicting
            print(loaded_model.predict(final_unique_decoded_sentences[i]))
            print(loaded_model.predict(final_unique_decoded_sentences[i]).flatten())

            preds[i] = loaded_model.predict(final_unique_decoded_sentences[i]).flatten()
        preds = preds[0]
    else:
        # Load the TF-IDF vectorizer of the respective dataset
        vectorizer = pickle.load(open(vect_filename, 'rb'))
        test_vectors = [[] for _ in range(number_of_sentences)]

        for i in range(number_of_sentences):
            final_unique_state_sentences[i].append(latent_space_state[i])
            final_unique_decoded_sentences[i].append(in_sentences[i])

            final_unique_state_sentences[i].extend(generated_state_sentences[i])
            final_unique_decoded_sentences[i].extend(generated_decoded_sentences[i])
            test_vectors[i] = vectorizer.transform(final_unique_decoded_sentences[i])
            # Predicting
            preds[i] = loaded_model.predict(test_vectors[i])

    return preds, final_unique_state_sentences, final_unique_decoded_sentences


def find_closest_k_latent_sentences(state_sentences, decoded_sentences, predictions, k):
    negative_distances = list()
    negative_idx_distances = list()
    positive_distances = list()
    positive_idx_distances = list()
    negative_state_sentences = list()
    positive_state_sentences = list()
    negative_decoded_sentences = list()
    positive_decoded_sentences = list()
    negative_predictions = list()
    positive_predictions = list()
    instance_state_sentence = state_sentences[0]
    instance_decoded_sentence = decoded_sentences[0]
    instance_prediction = predictions[0]

    for i in range(1, len(state_sentences)):
        if predictions[i] == 0:
            negative_state_sentences.append(state_sentences[i])
            negative_decoded_sentences.append(decoded_sentences[i])
            negative_predictions.append((predictions[i]))
        else:
            positive_state_sentences.append(state_sentences[i])
            positive_decoded_sentences.append(decoded_sentences[i])
            positive_predictions.append((predictions[i]))

    for i in range(len(negative_state_sentences)):
        negative_idx_distances.append(i)
        negative_distances.append(cdist(instance_state_sentence.reshape(1, -1),
                                        negative_state_sentences[i].reshape(1, -1), metric='euclidean').ravel())

    for i in range(len(positive_state_sentences)):
        positive_idx_distances.append(i)
        positive_distances.append(cdist(instance_state_sentence.reshape(1, -1),
                                        positive_state_sentences[i].reshape(1, -1), metric='euclidean').ravel())

    negative_distances_dict = dict(zip(negative_idx_distances, negative_distances))
    negative_distances_sorted = {k: v for k, v in sorted(negative_distances_dict.items(), key=lambda x: x[1])}

    negative_final_idxs, negative_final_dists = zip(*list(negative_distances_sorted.items()))
    negative_final_state_sentences = [negative_state_sentences[x] for x in negative_final_idxs[:100]]
    negative_final_decoded_sentences = [negative_decoded_sentences[x] for x in negative_final_idxs[:100]]
    negative_final_predictions = [negative_predictions[x] for x in negative_final_idxs[:100]]

    positive_distances_dict = dict(zip(positive_idx_distances, positive_distances))
    positive_distances_sorted = {k: v for k, v in sorted(positive_distances_dict.items(), key=lambda x: x[1])}

    positive_final_idxs, positive_final_dists = zip(*list(positive_distances_sorted.items()))
    positive_final_state_sentences = [positive_state_sentences[x] for x in positive_final_idxs[:100]]
    positive_final_decoded_sentences = [positive_decoded_sentences[x] for x in positive_final_idxs[:100]]
    positive_final_predictions = [positive_predictions[x] for x in positive_final_idxs[:100]]

    return [instance_state_sentence] + negative_final_state_sentences + positive_final_state_sentences, \
           [instance_decoded_sentence] + negative_final_decoded_sentences + positive_final_decoded_sentences, \
           [instance_prediction] + negative_final_predictions + positive_final_predictions


def default_kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


def calculate_weights(Z, metric):
    if np.max(Z) != 1 and np.min(Z) != 0:
        Zn = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
        distances = cdist(Zn, Zn[0].reshape(1, -1), metric=metric).ravel()
    else:
        distances = cdist(Z, Z[0].reshape(1, -1), metric=metric).ravel()

    weights = kernel(distances)
    return weights


def find_exemplars(Z, idxs, metric):
    distances = list()
    idx_distances = list()
    for t in idxs:
        distances.append(cdist(Z[0].reshape(1, -1), Z[t].reshape(1, -1), metric=metric).ravel())
        idx_distances.append(t)

    distances_dict = dict(zip(idx_distances, distances))
    distances_sorted = {k: v for k, v in sorted(distances_dict.items(), key=lambda x: x[1])}
    final_idxs, final_dists = zip(*list(distances_sorted.items()))

    return final_idxs[1:]


def find_counter_exemplars(Z, idxs, metric, count):
    distances = list()
    idx_distances = list()
    for t in idxs:
        distances.append(cdist(Z[0].reshape(1, -1), Z[t].reshape(1, -1), metric=metric).ravel())
        idx_distances.append(t)

    distances_dict = dict(zip(idx_distances, distances))
    distances_sorted = {k: v for k, v in sorted(distances_dict.items(), key=lambda x: x[1])}
    final_idxs, final_dists = zip(*list(distances_sorted.items()))
    return final_idxs[:count]


def EVAL_find_counter_exemplars(latent_representation_original, Z, idxs, counter_exemplar_idxs):
    """
    Compute the values of the goal function.

    """

    # prepare the data to apply the diversity optimization
    data = np.zeros((len(idxs), np.shape(Z)[1]))
    for i in range(len(idxs)):
        data[i] = Z[idxs[i]]

        # min-max normalization (applied on ALL examples)
    scaler = MinMaxScaler()
    scaler.fit_transform(data)

    # list of points
    points = [row for row in scaler.transform(data)]
    # MIN MAX normalize instance to explain
    instance = scaler.transform((latent_representation_original))

    # number of nearest neighbors to consider
    knn = 5

    kp = {}
    lconst = 1
    _, d0 = argmax(points, lambda p: -dist(instance, p))
    lconst = 0.5 / (-d0)

    for p1 in points:
        # compute distances
        dst = [(p2, dist(p1, p2)) for p2 in points if not np.array_equal(p1, p2)]
        # sort
        dst = sorted(dst, key=lambda x: x[1])
        # add top knn to kp
        kp[p1.tobytes()] = set(p2.tobytes() for p2, d in dst[:knn])

        # goal function

    def g(points):
        dpoints, dx = set(), 0
        for p1 in points:
            # union operator
            dpoints |= kp[p1.tobytes()]
            dx += dist(p1, instance)
        # scaled version 2*cost
        return len(dpoints) - 2 * lconst * dx

    # get the extracted CF
    extracted_CF_data = []
    for i in range(len(counter_exemplar_idxs)):
        extracted_CF_data.append(Z[counter_exemplar_idxs[i]])

        # apply scaling
    extracted_CF_data = scaler.transform((extracted_CF_data))

    return g(extracted_CF_data)


def DIVERSITY_find_counter_exemplars(latent_representation_original, Z, idxs, metric, count):
    """
    Pick CF based on diversity optimization.
    Additionally, the consecutive values of the goal function are returned
    """

    # prepare the data to apply the diversity optimization
    data = np.zeros((len(idxs), np.shape(Z)[1]))
    for i in range(len(idxs)):
        data[i] = Z[idxs[i]]

        # min-max normalization (applied on ALL examples)
    scaler = MinMaxScaler()
    scaler.fit_transform(data)

    # list of points
    points = [row for row in scaler.transform(data)]
    # MIN MAX normalize instance to explain
    instance = scaler.transform((latent_representation_original))

    # number of nearest neighbors to consider
    knn = 5

    kp = {}
    lconst = 1
    _, d0 = argmax(points, lambda p: -dist(instance, p))
    lconst = 0.5 / (-d0)

    for p1 in points:
        # compute distances
        dst = [(p2, dist(p1, p2)) for p2 in points if not np.array_equal(p1, p2)]
        # sort
        dst = sorted(dst, key=lambda x: x[1])
        # add top knn to kp
        kp[p1.tobytes()] = set(p2.tobytes() for p2, d in dst[:knn])

        # goal function

    def g(points):
        dpoints, dx = set(), 0
        for p1 in points:
            # union operator
            dpoints |= kp[p1.tobytes()]
            dx += dist(p1, instance)
        # scaled version 2*cost
        return len(dpoints) - 2 * lconst * dx

    gderiv = lambda p, x=None: g(p) if x is None else g(p + [x]) - g(p)

    # can use acover
    # cover holds the points chosen by the algorithm
    # g_values holds the consecutive values of the goal function
    covers, g_values = acover(points, df=gderiv, k=count, verbose=False)

    # from list back to matrix
    cov = np.concatenate([[a] for a in covers])
    # and rescale to original coordinates
    cov = scaler.inverse_transform(cov)

    # get back the indices of the elements that are chosen by the greedy algorithm

    data_indices = []

    for i in range(len(cov)):
        for j in range(len(data)):
            if np.allclose(cov[i], data[j]):
                data_indices.append(j)

    # convert into CF indices so that we can return a list that holds only CF elements, generated by greedy alg

    final_indices = []

    for i in range(len(data_indices)):
        final_indices.append(idxs[data_indices[i]])

    # get back the original data
    # print(Z[final_indices[i]])

    return final_indices, g_values


def find_most_common_words(A, n):
    split_words = ' '.join(A).split()

    stop_words = set(stopwords.words('english'))
    stop_words.add('<end>')
    stop_words.add('film')
    stop_words.add('movie')

    filtered_split_words = [w for w in split_words if not w in stop_words]

    try:
        top_n_words, top_n_words_count = zip(*Counter(filtered_split_words).most_common(n))
    except ValueError:
        return [], []
    top_n_words_relative_count = np.array(top_n_words_count) / len(filtered_split_words)

    return top_n_words, top_n_words_relative_count


def pickle_dump_files():
    """
    Saves predictions, state_sentences, decoded_sentences, id,
    exemplars, counter exemplars along with top words for instability experiment
    :return:
    """
    with open('data/' + dataset_name + '_' + model_name + '_' + 'predictions', 'wb') as f:
        pickle.dump(predictions, f)
    with open('data/' + dataset_name + '_' + model_name + '_' + 'state_sentences', 'wb') as f:
        pickle.dump(final_state_sentences, f)
    with open('data/' + dataset_name + '_' + model_name + '_' + 'decoded_sentences', 'wb') as f:
        pickle.dump(final_decoded_sentences, f)
    with open('data/' + dataset_name + '_' + model_name + '_' + 'ids', 'wb') as f:
        pickle.dump(idx, f)
    with open('data/' + dataset_name + '_' + model_name + '_' + 'exemplars', 'wb') as f:
        pickle.dump(exemplars, f)
    with open('data/' + dataset_name + '_' + model_name + '_' + 'counter_exemplars', 'wb') as f:
        pickle.dump(counter_exemplars, f)
    with open('data/' + dataset_name + '_' + model_name + '_' + 'top_exemplar_words', 'wb') as f:
        pickle.dump(top_exemplar_words, f)
    with open('data/' + dataset_name + '_' + model_name + '_' + 'top_counter_exemplar_words', 'wb') as f:
        pickle.dump(top_counter_exemplar_words, f)


def create_explanations_csv():
    """
    Creates csv with final explanations along with other results from X-SPELLS run
    :return:
    """
    with open('output/' + dataset_name + '_' + model_name + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ["index", "original text", "true class", "decoded text", "black box prediction",
             "decision tree prediction", "fidelity", "exemplars", "counter exemplars", "top exemplar words",
             "top counter exemplar words"])
        for i in range(len(idx)):
            writer.writerow(
                [idx[i], X_original[i], y_original[i], final_decoded_sentences[i][0], bbpreds[i], dtpreds[i],
                 fidelities[i], exemplars[i], counter_exemplars[i], top_exemplar_words_dict_list[i],
                 top_counter_exemplar_words_dict_list[i]])


if __name__ == "__main__":
    # Initialize stuff
    # Insert 'hate' or 'polarity' as dataset
    dataset_name = "liar"
    # Insert 'RF' or 'DNN' as black box model
    model_name = "DNN"
    pickled_black_box_filename = 'models/' + dataset_name + '_saved_' + model_name + '_model.sav'

    if model_name == "RF":
        pickled_vectorizer_filename = 'models/' + dataset_name + '_tfidf_vectorizer.pickle'
    elif model_name == "DNN":
        # Insert None in vectorizer if black box is DNN
        pickled_vectorizer_filename = None

    mode = "DIVERSITY"  # DIVERSITY OR DISTANCE

    # For how many sentences we want to run X-SPELLS
    no_of_sentences = 1
    latent_dim = 500
    nbr_features = latent_dim

    if dataset_name == "youtube":
        res = YOUTUBE_get_text_data(num_samples=20000,
                                    data_path='data/YouTube-Spam-Collection-v1/' + dataset_name + '.csv',
                                    dataset=dataset_name)
    else:
        res = get_text_data(num_samples=20000, dataset=dataset_name)

    max_encoder_seq_length, num_enc_tokens, characters, char2id, id2char, \
    encoder_input_data, decoder_input_data, input_texts_original, X_original, y_original, X_original_processed = res
    input_dim = encoder_input_data.shape[-1]

    vae, enc, gen, stepper = load_VAE(dataset_name)
    # calculate_MRE()

    in_sentences, latent_space_state, decoded_sentences = get_sentences()
    smallest_x, largest_x = calculate_min_max(np.array(latent_space_state))

    generated_state_sentences, generated_decoded_sentences = generate_sentences(number_of_sentences=no_of_sentences,
                                                                                number_of_max_attempts=5000,
                                                                                number_of_random_sentences=200,
                                                                                probability=0.4)

    print(pickled_black_box_filename)
    predictions, final_state_sentences, final_decoded_sentences = get_predictions(pickled_black_box_filename,
                                                                                  pickled_vectorizer_filename,
                                                                                  no_of_sentences)

    # Initialize lists for later use
    idx, fidelities, bbpreds, dtpreds, exemplars, counter_exemplars, top_exemplar_words, top_counter_exemplar_words, \
    top_exemplar_words_dict_list, top_counter_exemplar_words_dict_list = ([] for i in range(10))

    for i in range(len(predictions)):
        print(i)
        y = list()

        if len(final_decoded_sentences[i]) < 40:
            print(len(final_decoded_sentences[i]))
            print('Not enough random sentences.')
            continue

        class_imbalance = False

        Z = np.array(final_state_sentences[i]).squeeze()  # convert from 3d to 2d

        Z_text = final_decoded_sentences[i]
        Yb = np.array(predictions[i])
        Z, Z_text, Yb = find_closest_k_latent_sentences(Z, Z_text, Yb, 100)
        Z = np.array(Z)
        Yb = np.array(Yb)

        exemplars_holder = list()
        counter_exemplars_holder = list()

        Y_0 = (np.count_nonzero(Yb == 0))
        Y_1 = (np.count_nonzero(Yb == 1))

        # Define as having an imbalance problem when either one of two classes has less than 40% of the total examples
        if Y_0 / (Y_0 + Y_1) < 0.4 or Y_1 / (Y_0 + Y_1) < 0.4:
            class_imbalance = True

        # Catch SMOTE error
        if Y_0 < 6 or Y_1 < 6:
            print('Not enough samples for smote.')
            continue

        # If we have class imbalance, apply SMOTE
        if class_imbalance:
            sm = SMOTE(random_state=42)
            Z, Yb = sm.fit_resample(Z, Yb)

        for t in range(len(Z)):
            y.append(np.expand_dims(Z[t], axis=0))  # convert from 2d to 3d

        for t in range(len(Z_text), len(Z)):
            Z_text.append(decode(y[t]))

        # Selecting a percentage to test fidelity on the dt
        indices = np.random.permutation(len(Z))
        Z_train_size = 0.95
        Z_test_size = 0.05

        Z_train, Z_test = Z[indices[:int(len(Z) * Z_train_size)]], Z[indices[int(len(Z) * Z_train_size):]]
        Yb_train, Yb_test = Yb[indices[:int(len(Z) * Z_train_size)]], Yb[indices[int(len(Z) * Z_train_size):]]

        # Calculate weights
        metric = 'euclidean'  # 'euclidean'
        kernel_width = float(np.sqrt(nbr_features) * 0.75)
        kernel = default_kernel
        kernel = partial(kernel, kernel_width=kernel_width)
        weights = calculate_weights(Z_train, metric)

        # Train latent decision tree
        class_values = ['0', '1']
        dt = decision_tree.learn_local_decision_tree(Z_train, Yb_train, weights, class_values, prune_tree=False)
        Yc = dt.predict(Z)
        print('Yc: ', Yc)

        opposite_prediction_idx = list()
        for t in range(len(Yc)):
            # We want the opposite of the instance's prediction
            if Yc[0] == 0:
                opposite_prediction_idx = np.where(Yc == 1)[0]
            else:
                opposite_prediction_idx = np.where(Yc == 0)[0]

        print('opposite_prediction_idx: ', opposite_prediction_idx)

        nbr_exemplars = 5
        if mode == "DIVERSITY":
            counter_exemplar_idxs, function_value = DIVERSITY_find_counter_exemplars(latent_space_state[i], Z,
                                                                                     opposite_prediction_idx,
                                                                                     metric='cosine',
                                                                                     count=nbr_exemplars)
            # print(function_value)
            print("final goal value (DIVERSITY)", function_value[len(function_value) - 1])
            function_value = function_value[len(function_value) - 1]
        if mode == "DISTANCE":
            counter_exemplar_idxs = find_counter_exemplars(Z, opposite_prediction_idx, metric='cosine',
                                                           count=nbr_exemplars)
            function_value = EVAL_find_counter_exemplars(latent_space_state[i], Z, opposite_prediction_idx,
                                                         counter_exemplar_idxs)
            print("final goal value (DISTANCE)", function_value)

        print(counter_exemplar_idxs)

        leave_id = dt.apply(Z)
        print('leave id: ', leave_id)
        others_in_same_leaf = np.where(leave_id == leave_id[0])[0]
        print('others in same leaf: ', others_in_same_leaf)
        print('original sentence: ', Z_text[0])

        if len(others_in_same_leaf) < nbr_exemplars:
            print('Not enough exemplars in the leaf, will find by distance instead...', len(others_in_same_leaf))
            same_prediction_idx = list()
            for t in range(1, len(Yc)):
                # We want the same as the instance's prediction
                if Yc[0] == 0:
                    same_prediction_idx = np.where(Yc == 0)[0]
                else:
                    same_prediction_idx = np.where(Yc == 1)[0]

            unique_exemplars = list(set(find_exemplars(Z, same_prediction_idx, metric='cosine')))
            print(unique_exemplars)
            selected_exemplars = unique_exemplars[:nbr_exemplars]
        else:
            selected_exemplars = np.random.choice(others_in_same_leaf, size=nbr_exemplars, replace=False)

        number_of_words = 5
        print('exemplars:')
        for j in selected_exemplars:
            print(Z_text[j])
            exemplars_holder.append(Z_text[j])

        np_exemplars_holder = np.array(exemplars_holder)
        top_n_exemplar_words, top_n_exemplar_words_relative_count = \
            find_most_common_words(np_exemplars_holder, number_of_words)

        print(top_n_exemplar_words)
        print(top_n_exemplar_words_relative_count)

        top_exemplar_words_dict = dict(zip(top_n_exemplar_words, top_n_exemplar_words_relative_count))

        print(top_exemplar_words_dict)

        print('counter exemplars:')
        for j in counter_exemplar_idxs:
            print(Z_text[j])
            counter_exemplars_holder.append(Z_text[j])

        np_counter_exemplars_holder = np.array(counter_exemplars_holder)
        top_n_counter_exemplar_words, top_n_counter_exemplar_words_relative_count = \
            find_most_common_words(np_counter_exemplars_holder, number_of_words)

        print(top_n_counter_exemplar_words)
        print(top_n_counter_exemplar_words_relative_count)

        top_counter_exemplar_words_dict = dict(zip(top_n_counter_exemplar_words,
                                                   top_n_counter_exemplar_words_relative_count))

        print(top_counter_exemplar_words_dict)
        print('original sentence', X_original[i])
        print('true class', y_original[i])
        print('black box prediction', Yb[0])
        print('decision tree prediction', Yc[0])

        fidelity = accuracy_score(Yb, Yc)
        print('fidelity', fidelity)

        idx.append(i)
        fidelities.append(fidelity)
        bbpreds.append(Yb[0])
        dtpreds.append(Yc[0])
        exemplars.append(exemplars_holder)
        counter_exemplars.append(counter_exemplars_holder)
        top_exemplar_words.append(top_n_exemplar_words)
        top_counter_exemplar_words.append(top_n_counter_exemplar_words)
        top_exemplar_words_dict_list.append(top_exemplar_words_dict)
        top_counter_exemplar_words_dict_list.append(top_counter_exemplar_words_dict)
        print('')

    pickle_dump_files()
    create_explanations_csv()

"""
Trains the VAE model and saves it inside models for later use
"""
import os
import string
import sys
from collections import Counter

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

sys.path.insert(0, '..')

from lstm_vae import create_lstm_vae, inference
from preprocessing.pre_processing import preProcessing

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


def get_text_data(num_samples, dataset):
    thousandwords = [line.rstrip('\n') for line in open(os.path.join(__location__, '../data/1-1000.txt'))]

    print('thousandwords', thousandwords)
    # vectorize the data
    input_texts = []
    input_texts_original = []
    input_words = set(["\t"])
    all_input_words = []
    lines = []

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

    # we use "best possible case" only i.e. only test data
    new_X_test = preProcessing(X_test)

    print(len(X_test))

    for line in new_X_test:
        input_texts_original.append(line)
        lines.append(
            line.lower().translate(str.maketrans('', '', string.punctuation)))  # lowercase and remove punctuation
    print(lines)

    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text = line

        input_text = word_tokenize(input_text)
        input_text.append("<end>")
        input_texts.append(input_text)

        for word in input_text:
            if word not in input_words:
                input_words.add(word)

        for word in input_text:  # This will be used to count the words and keep the most frequent ones
            all_input_words.append(word)

    words_to_keep = 4999
    most_common_words = [word for word, word_count in
                         Counter(all_input_words).most_common(words_to_keep)]  # Keep the 1000 most common words
    most_common_words.append('\t')

    for word in thousandwords:  # Here we add the 1000 most common english words
        most_common_words.append(word)

    print(most_common_words)

    input_texts_cleaned = [[word for word in text if word in most_common_words] for text in input_texts]

    print(len(input_texts_cleaned))

    final_input_words = sorted(list(set(most_common_words)))
    num_encoder_tokens = len(final_input_words)
    max_encoder_seq_length = max([len(txt) for txt in input_texts_cleaned]) + 1

    print("input_texts_cleaned", input_texts_cleaned)
    print(most_common_words)
    print(final_input_words)

    print("Number of samples:", len(input_texts_cleaned))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)

    input_token_index = dict([(char, i) for i, char in enumerate(final_input_words)])
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())

    encoder_input_data = np.zeros((len(input_texts_cleaned), max_encoder_seq_length, num_encoder_tokens),
                                  dtype="float32")
    decoder_input_data = np.zeros((len(input_texts_cleaned), max_encoder_seq_length, num_encoder_tokens),
                                  dtype="float32")

    for i, input_text_cleand in enumerate(input_texts_cleaned):
        decoder_input_data[i, 0, input_token_index["\t"]] = 1.0

        for t, char in enumerate(input_text_cleand):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
            decoder_input_data[i, t + 1, input_token_index[char]] = 1.0

    print('.......')
    for i in range(10):
        print(input_texts[i])
        print(input_texts_cleaned[i])
        print('')

    return max_encoder_seq_length, num_encoder_tokens, final_input_words, input_token_index, reverse_input_char_index, \
           encoder_input_data, decoder_input_data, input_texts_original, X_test, y_test, new_X_test


def decode(s):
    return inference.decode_sequence(s, gen, stepper, input_dim, char2id, id2char, max_encoder_seq_length)


if __name__ == "__main__":
    dataset_name = 'liar'
    res = get_text_data(num_samples=20000, dataset=dataset_name)

    max_encoder_seq_length, num_enc_tokens, characters, char2id, id2char, \
    encoder_input_data, decoder_input_data, input_texts_original, X_original, y_original, X_original_processed = res

    print(encoder_input_data.shape, "Creating model...")

    input_dim = encoder_input_data.shape[-1]
    batch_size = 1
    latent_dim = 500
    intermediate_dim = 256

    if dataset_name == 'hate':
        epochs = 200
    elif dataset_name == 'polarity':
        epochs = 250
    elif dataset_name == 'liar':
        epochs = 300

    vae, enc, gen, stepper, vae_loss = create_lstm_vae(input_dim,
                                                       batch_size=batch_size,
                                                       intermediate_dim=intermediate_dim,
                                                       latent_dim=latent_dim)
    print("Training VAE model...")

    vae.fit([encoder_input_data, decoder_input_data], encoder_input_data, epochs=epochs, verbose=1)
    vae.save('../models/' + dataset_name + '_vae_model.h5', overwrite=True)
    enc.save('../models/' + dataset_name + '_enc_model.h5', overwrite=True)
    gen.save('../models/' + dataset_name + '_gen_model.h5', overwrite=True)
    stepper.save('../models/' + dataset_name + '_stepper_model.h5', overwrite=True)

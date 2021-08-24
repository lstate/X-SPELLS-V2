#!/usr/bin/env python
# coding: utf-8

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


# updated function
# additional split into test and training set
# most common words only extracted from the training set
# num encoder seq length: training set and 0.1 "safety margin"

# parameter num_samples: can we omit it

def get_text_data(num_samples, data_path, dataset):
    thousandwords = [line.rstrip('\n') for line in open('../data/1-1000.txt')]

    print('thousandwords', thousandwords)
    # vectorize the data
    input_texts = []
    input_texts_test = []
    input_texts_original = []
    input_texts_original_test = []
    
    input_words = set(["\t"])
    all_input_words = []
    
    lines = []
    lines_test = []
    
    df = pd.read_csv(data_path, encoding='utf-8')

    if dataset == "polarity":
        X = df['tweet'].values
        y = df['class'].values
    elif dataset == "hate":
        # Removing the offensive comments, keeping only neutral and hatespeech,
        # and convert the class value from 2 to 1 for simplification purposes
        df = df[df['class'] != 1]
        X = df['tweet'].values
        y = df['class'].apply(lambda x: 1 if x == 2 else 0).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.25)
    
    # add another split of the data set
    # parameter stratify: preserves class-relations in data set    
    X_train_subsplit, X_test_subsplit, y_train_subsplit, y_test_subsplit = train_test_split(X_test, y_test, random_state=42, stratify=y_test, test_size=0.25)

    # new_X_test = preProcessing(X_test)
    new_X_train_subsplit = preProcessing(X_train_subsplit)
    new_X_test_subsplit = preProcessing(X_test_subsplit)
    
    # clean training set
    for line in new_X_train_subsplit:
        input_texts_original.append(line)
        # lowercase and remove punctuation
        lines.append(line.lower().translate(str.maketrans('', '', string.punctuation)))  
    
    # clean test set
    for line in new_X_test_subsplit:
        input_texts_original_test.append(line)
        # lowercase and remove punctuation
        lines_test.append(line.lower().translate(str.maketrans('', '', string.punctuation)))  
    
    # create training set where all lines are in a single text file, divided by <end>
    # create a single list with all words (they can appear multiple times)
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text = line
        input_text = word_tokenize(input_text)
        input_text.append("<end>")
        
        input_texts.append(input_text)

        # do we need this section ?
        for word in input_text:
            if word not in input_words:
                input_words.add(word)

        for word in input_text:  
            # This will be used to count the words and keep the most frequent ones
            all_input_words.append(word)
            
    # create test set where all lines are in a single text file, as above
    for line in lines_test[: min(num_samples, len(lines_test) - 1)]:
        input_text = line
        input_text = word_tokenize(input_text)
        input_text.append("<end>")
        
        input_texts_test.append(input_text)

    # Keep the 5000 most common words only
    words_to_keep = 4999
    most_common_words = [word for word, word_count in Counter(all_input_words).most_common(words_to_keep)]  
    most_common_words.append('\t')
    
    # Add the 1000 most common english words
    for word in thousandwords:  
        most_common_words.append(word)

    # cleaned texts only for X_train_subsplit
    input_texts_cleaned = [[word for word in text if word in most_common_words] for text in input_texts]
    input_texts_cleaned_test = [[word for word in text if word in most_common_words] for text in input_texts_test]
    
    final_input_words = sorted(list(set(most_common_words)))
    num_encoder_tokens = len(final_input_words)
    # final input words is longer than 5000 + 1000: sorting in python
    
    # max encoder seq length 
    max_encoder_seq_length = max([len(txt) for txt in input_texts_cleaned]) + 1
    # since the max length can be different for unseen examples, a "safety margin" of 0.1 is added
    max_encoder_seq_length = int (max_encoder_seq_length * 1.1)
    
    print("input_texts_cleaned", input_texts_cleaned)
    #print(most_common_words)
    #print(final_input_words)

    print("Number of samples (train):", len(input_texts_cleaned))
    print("Number of samples (test):", len(input_texts_cleaned_test))
    print("Number of unique input tokens:", num_encoder_tokens)
    
    # output is different (and also different to paper!)
    print("Max sequence length for inputs:", max_encoder_seq_length)

    input_token_index = dict([(char, i) for i, char in enumerate(final_input_words)])
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())

    encoder_input_data = np.zeros((len(input_texts_cleaned), max_encoder_seq_length, num_encoder_tokens),
                                  dtype="float32")
    decoder_input_data = np.zeros((len(input_texts_cleaned), max_encoder_seq_length, num_encoder_tokens),
                                  dtype="float32")
    
    encoder_input_data_test = np.zeros((len(input_texts_cleaned_test), max_encoder_seq_length, num_encoder_tokens),
                                  dtype="float32")
    decoder_input_data_test = np.zeros((len(input_texts_cleaned_test), max_encoder_seq_length, num_encoder_tokens),
                                  dtype="float32")

    for i, input_text_cleand in enumerate(input_texts_cleaned):
        # index and a single line from the whole text list, ended by <edn>
        # write a marker for the "stop sign" at the first position of each line ?!
        decoder_input_data[i, 0, input_token_index["\t"]] = 1.0

        for t, char in enumerate(input_text_cleand):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
            decoder_input_data[i, t + 1, input_token_index[char]] = 1.0
            
    for i, input_text_cleand in enumerate(input_texts_cleaned_test):
        # index and a single line from the whole text list, ended by <edn>
        # write a marker for the "stop sign" at the first position of each line ?!
        decoder_input_data_test[i, 0, input_token_index["\t"]] = 1.0

        for t, char in enumerate(input_text_cleand):
            encoder_input_data_test[i, t, input_token_index[char]] = 1.0
            decoder_input_data_test[i, t + 1, input_token_index[char]] = 1.0

    return max_encoder_seq_length, num_encoder_tokens, final_input_words, input_token_index, reverse_input_char_index,            encoder_input_data, decoder_input_data, input_texts_original, X_test, y_test, X_train_subsplit, X_test_subsplit, y_train_subsplit, y_test_subsplit,            encoder_input_data_test, decoder_input_data_test, input_texts_original_test, new_X_train_subsplit, new_X_test_subsplit

def decode(s):
    return inference.decode_sequence(s, gen, stepper, input_dim, char2id, id2char, max_encoder_seq_length)

# ---

dataset_name = 'hate'

res = get_text_data(num_samples=20000, data_path='../data/' + dataset_name + '_tweets.csv', dataset=dataset_name)

max_encoder_seq_length, num_enc_tokens, characters, char2id, id2char, encoder_input_data_train, decoder_input_data_train, input_texts_original, X_, y_, X_trains_sub, X_test_sub, y_train_sub, y_test_sub, encoder_input_data_test, decoder_input_data_test, input_texts_original_test, new_X_train_subsplit, new_X_test_subsplit = res

print(encoder_input_data_train.shape, "Creating model...")

# number of unique input tokens (number of unique words)
input_dim = encoder_input_data_train.shape[-1]

batch_size = 1
latent_dim = 500
intermediate_dim = 256
epochs = 100

vae, enc, gen, stepper, vae_loss = create_lstm_vae(input_dim,
                                                   batch_size=batch_size,
                                                   intermediate_dim=intermediate_dim,
                                                   latent_dim=latent_dim)

print("Training VAE model...")

vae.fit([encoder_input_data_train, decoder_input_data_train], encoder_input_data_train, epochs=epochs, verbose=1,
        validation_data=([encoder_input_data_test, decoder_input_data_test], encoder_input_data_test))

np.save('vae_training_history_hate_100_75_25.npy', vae.history.history)

# safe the trained autoencoder

vae.save('../models/' + dataset_name + '_vae_model_hate_100_75_25.h5', overwrite=True)
enc.save('../models/' + dataset_name + '_enc_model_hate_100_75_25.h5', overwrite=True)
gen.save('../models/' + dataset_name + '_gen_model_hate_100_75_25.h5', overwrite=True)
stepper.save('../models/' + dataset_name + '_stepper_model_hate_100_75_25.h5', overwrite=True)

print("successfully trained + saved")

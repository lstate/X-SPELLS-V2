"""
Handles the pre-processing used across various other scripts
"""

import re
import string

import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def cleanText(var):
    # replace punctuation with spaces
    var = re.sub('[{}]'.format(string.punctuation), " ", var)
    # remove double spaces
    var = re.sub(r'\s+', " ", var)
    # put in lower case
    var = var.lower().split()
    # remove words that are smaller than 3 characters
    var = [w for w in var if len(w) >= 3]
    # remove stop-words
    # var = [w for w in var if w not in stopwords.words('english')]
    # stemming
    # stemmer = nltk.PorterStemmer()
    # var = [stemmer.stem(w) for w in var]
    var = " ".join(var)
    return var


# Removes 'rt' from all input data
def my_clean(text):
    text = text.lower().split()
    text = [w for w in text]
    text = " ".join(text)
    text = re.sub(r"rt", "", text)
    return text

# Removes 'rt' from all input data
# Removes emojis from all input data
def YOUTUBE_my_clean(text):
    text = text.lower().split()
    text = [w for w in text]
    text = " ".join(text)
    text = re.sub(r"rt", "", text)
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = re.sub(emoji_pattern, '', text)
    return text


def strip_links(text):
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text


def strip_all_entities(text):
    entity_prefixes = ['@', '#']
    for separator in string.punctuation:
        if separator not in entity_prefixes:
            text = text.replace(separator, ' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


def preProcessing(strings):
    clean_tweet_texts = []
    for string in strings:
        clean_tweet_texts.append(my_clean(strip_all_entities(strip_links(string))))
        # clean_tweet_texts.append(my_clean(string))
    return clean_tweet_texts

def YOUTUBE_preProcessing(strings):
    clean_tweet_texts = []
    for string in strings:
        clean_tweet_texts.append(YOUTUBE_my_clean(strip_all_entities(strip_links(string))))
        # clean_tweet_texts.append(my_clean(string))
    return clean_tweet_texts

def get_text_data(data_path, dataset):
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
    #elif dataset == "youtube":
    #    X = df["CONTENT"].values
    #    y = df["CLASS"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.25)
    
    new_X_train = preProcessing(X_train)
    new_X_test = preProcessing(X_test)
    
    #print(new_X_train)
    #for i in range(len(new_X_test)):
    #    print(new_X_test[i])
        
    print(len(new_X_train))
    print(len(y_train))
    
    print(len(new_X_test))
    print(len(y_test))
    
    return X_train, X_test, y_train, y_test, new_X_train, new_X_test

def YOUTUBE_get_text_data(data_path, dataset):
    df = pd.read_csv(data_path, encoding='utf-8')

    X = df["CONTENT"].values
    y = df["CLASS"].values
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.25)

    #new_X_test = X_test
    #new_X_train = X_train
    
    new_X_train = YOUTUBE_preProcessing(X_train)
    new_X_test = YOUTUBE_preProcessing(X_test)
    
    # delete x/y where there is no more content after preprocessing or we have more than 140 characters (e.g. comment was only an url)
    
    indx = []
    for i in range(len(new_X_test)):
        if len(new_X_test[i]) == 0:
            indx.append(i)
        elif len(new_X_test[i]) > 140:
            indx.append(i)     
    new_X_test = np.delete(new_X_test, indx, 0)
    y_test = np.delete(y_test, indx, 0)
    
    indx_train = []
    for i in range(len(new_X_train)):
        if len(new_X_train[i]) == 0:
            indx_train.append(i)
        if len(new_X_train[i]) > 140:
            indx_train.append(i)
    #new_X_train = np.delete(new_X_train, indx_train, 0)
    #y_train = np.delete(y_train, indx_train, 0)
    
    new_X_train = np.delete(new_X_train, indx, 0)
    y_train = np.delete(y_train, indx, 0)

    return X_train, X_test, y_train, y_test, new_X_train, new_X_test

def word_count(data_path, dataset):
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

    wordcounts = list()
    for sentence in X:
        # with nltk tokenize
        nltk_tokens = nltk.word_tokenize(sentence)
        # naive way, splitting words by spaces
        naive_words = sentence.split(' ')
        print(nltk_tokens)
        print(naive_words)
        wordcounts.append(len(nltk_tokens))

    print('\n')
    average_wordcount = sum(wordcounts) / len(wordcounts)
    no_tweets = len(wordcounts)

    return no_tweets, average_wordcount

#!/usr/bin/env python

""" TODO: This Python script trains Naive Bayes, Decision Tree, Random Forest and K-nearest Neighbors
    classifier for multi-class classification on text data set, and prints some predictive analytics. """

import argparse
import random
import pandas as pd
import numpy as np
import time
import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# import string.punctuation
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import scipy.sparse as sp

nlp = spacy.load("en_core_web_sm")


def create_arg_parser():
    parser = argparse.ArgumentParser()
    """ Below argument is used for parameter tuning purposes (to generate train, dev and test set from reviews.txt) """
    parser.add_argument("-i", "--input_file", default='reviews.txt', type=str,
                        help="Training set file to learn from (default reviews.txt)")
    # parser.add_argument("-i", "--trainset", default='trainset.txt', type=str,
    #                     help="Training set file to learn from (default trainset.txt)")
    # parser.add_argument("-ts", "--testset", default='testset.txt', type=str,
    #                     help="Test set file to learn from (default testset.txt)")
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-cr", "--create_custom_features", action="store_true",
                        help="Create custom feature matrix and train the svm model")
    """ Below argument is used for split_data function (to generate train, dev and test set from reviews.txt) """
    parser.add_argument("-tp", "--test_percentage", default=0.15, type=float,
                        help="Percentage of the data that is used for the test set (default 0.15)")
    """ Below argument is used for split_data function (to generate train, dev and test set from reviews.txt) """
    parser.add_argument("-sh", "--shuffle", action="store_true",
                        help="Shuffle data set before splitting in train/test")
    args = parser.parse_args()
    return args


# Removing Punctuation, Stop words and applying Porter Stemmer
ps = PorterStemmer()


def preprocess(x):
    doc = nlp(x)
    processed = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return processed


def spacy_pos(x):
    doc = nlp(x)
    cleaned_doc = [token.lemma_ + '_' + token.pos_ for token in doc if not token.is_stop and not token.is_punct]
    return cleaned_doc


# def read_corpus(corpus_file):
#     """ This function reads the txt file, and seperates labels and text IDs from each text item/line.
#     It returns text/tweet, its label (dvd, music etc., and text IDs (530.txt etc.)) """
#     documents = []
#     labels = []
#     ids = []
#     with open(corpus_file, encoding='utf-8') as f:
#         for line in f:
#             tokens = line.strip().split()
#             documents.append(tokens[3:])
#             # 2-class problem: neg, pos
#             ids.append(tokens[2])
#             labels.append(tokens[1])
#     return documents, labels, ids


def read_corpus(corpus_file):
    documents = []
    labels = []
    ids = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in tqdm(f):
            tokens = line.strip().split()
            documents.append(' '.join(tokens[3:]))
            ids.append(tokens[2])
            labels.append(tokens[1])
    return documents, labels, ids


def shuffle_dependent_lists(l1, l2, l3):
    """ Shuffle two lists, but keep the dependency between them. """
    tmp = list(zip(l1, l2, l3))
    # Seed the random generator so results are consistent between runs
    random.Random(123).shuffle(tmp)
    return zip(*tmp)


def split_data(X_full, Y_full, ids, test_percentage, shuffle):
    """ Splits the data set as training (70% from beginning of the file), dev set (% after split point)
        and test set (rest of the data after assigning train and dev set).
        It returns training, development and test set texts, and their labels, and their IDs. """
    split_point = int((1.0 - test_percentage * 2) * len(X_full))
    split_point_test = int(test_percentage * len(X_full))
    # It shuffles the data set before splitting the data if shuffle returns True.
    if shuffle:
        X_full, Y_full, ids = shuffle_dependent_lists(X_full, Y_full, ids)
    X_train = X_full[:split_point]
    Y_train = Y_full[:split_point]
    ids_train = ids[:split_point]
    X_dev = X_full[split_point:split_point + split_point_test]
    Y_dev = Y_full[split_point:split_point + split_point_test]
    ids_dev = ids[split_point:split_point + split_point_test]
    X_test = X_full[split_point + split_point_test:]
    Y_test = Y_full[split_point + split_point_test:]
    ids_test = ids[split_point + split_point_test:]
    return X_train, Y_train, ids_train, X_dev, Y_dev, ids_dev, X_test, Y_test, ids_test


def identity(x):
    """ Dummy function that just returns the input. """
    return x


def get_confusion_matrix(Y_test, Y_pred, clf):
    """ Get the confusion matrix. """

    result = confusion_matrix(Y_test, Y_pred, labels=clf.classes_)
    result_panda = pd.DataFrame(list(result), columns=clf.classes_, index=clf.classes_)

    print("Confusion Matrix: \n")
    return result_panda


def num_capitalized(txt):
    return sum([1 for tok in txt.split() if tok[0].isupper() and tok[0].isalpha()])


def num_tokens(txt):
    return sum([1 for tok in txt.split()])


def num_full_capitals(txt):
    return sum([1 for tok in txt.split() if tok.isupper() and tok.isalpha()])


def avg_word_length(txt):
    return sum([len(tok) for tok in txt.split() if tok[0].isupper()]) / len(txt)


def create_feature_dict(txt):
    dic = dict()
    dic["num_tokens"] = num_tokens(txt)
    dic["num_capitalized"] = num_capitalized(txt)
    dic["num_full_cap"] = num_full_capitals(txt)
    dic["avg_word_len"] = avg_word_length(txt)
    return dic


def create_custom_features(X_train):
    # Extract features for each text:
    X_train_dict = [create_feature_dict(txt) for txt in X_train]
    # Create the vectorizer and immediately transform to matrix format
    word_matrix = CountVectorizer(tokenizer=identity, preprocessor=identity).fit_transform(X_train)
    dic_matrix = DictVectorizer().fit_transform(X_train_dict)
    # Concatenate the two sparse matrices by row
    X_train_matrix = sp.hstack((word_matrix, dic_matrix), format='csr')
    return X_train_matrix


def svm_svc(vec, X_train, Y_train, X_test, Y_test, kernel="linear", C=1.0, gamma=0.7):
    if kernel == "rbf":
        # Combine the vectorizer with a SVC classifier
        clf = Pipeline([('vec', vec), ('cls', SVC(kernel="rbf", C=C, gamma=gamma))])
    else:
        clf = Pipeline([('vec', vec), ('cls', SVC(kernel=kernel, C=C))])

    t0 = time.time()
    clf.fit(X_train, Y_train)
    train_time = time.time() - t0
    print("Training time: ", train_time)

    # Given a trained model, predict the label of a new set of data.

    t = time.time()
    Y_pred = clf.predict(X_test)
    test_time = time.time() - t
    print("Testing time: ", test_time)

    # Calculates the accuracy score of the trained model by comparing predicted labels with actual labels.
    acc = accuracy_score(Y_test, Y_pred)
    # Calculating the cross validation scores
    # pred = cross_val_predict(clf, X_train, Y_train, cv=5)
    # print("\nClassification report for cross-validation: \n {}".format(classification_report(Y_train, pred, digits=3)))

    print("\nFinal accuracy: {}".format(acc))
    print(get_confusion_matrix(Y_test, Y_pred, clf))
    print("\nClassification report: \n {}".format(classification_report(Y_test, Y_pred, target_names=clf.classes_,
                                                                        digits=3)))


def svm_linear_svc(vec, X_train, Y_train, X_test, Y_test, C=1.0):
    clf = Pipeline([('vec', vec), ('cls', LinearSVC(random_state=42, C=C))])

    t0 = time.time()
    clf.fit(X_train, Y_train)
    train_time = time.time() - t0
    print("Training time: ", train_time)

    t = time.time()
    Y_pred = clf.predict(X_test)
    test_time = time.time() - t
    print("Testing time: ", test_time)

    acc = accuracy_score(Y_test, Y_pred)
    # Calculating the cross validation scores
    # pred = cross_val_predict(clf, X_train, Y_train, cv=5)
    # print("\nClassification report for cross-validation: \n {}".format(classification_report(Y_train, pred, digits=3)))

    print("\nFinal accuracy: {}".format(acc))
    print(get_confusion_matrix(Y_test, Y_pred, clf))
    print("\nClassification report: \n {}".format(classification_report(Y_test, Y_pred, target_names=clf.classes_,
                                                                        digits=3)))


def best_svm():
    return


if __name__ == "__main__":
    args = create_arg_parser()

    """ Below code refactored for the format python LFD_assignment2.py -i <trainset> -ts <testset>.
        Normally, it is used with split_data function to experiment with different classifiers. """

    X_full, Y_full, ids = read_corpus(args.input_file)

    """ Below code used to generate development set to tune model parameters. """

    X_train, Y_train, ids_train, X_dev, Y_dev, ids_dev, X_test, Y_test, ids_test = split_data(X_full, Y_full, ids,
                                                                                              args.test_percentage,
                                                                                              args.shuffle)

    """ Below code used for splitting data to test the trained models after tuning the parameters."""
    # X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.20, random_state=42)

    # Reads the text file and and return texts and their labels by tokenizing them, for both training
    # and test data set.

    # X_train, Y_train, ids = read_corpus(args.trainset)
    # X_test, Y_test, ids = read_corpus(args.testset)

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.

    if args.create_custom_features:
        print("-----SVC with custom features-----")
        # Union of features is already done, pipeline is just the algorithm
        clf = Pipeline([('cls', SVC(kernel="linear", C=1.0))])
        # clf = Pipeline([('cls', SVC(kernel="rbf", gamma=0.7, C=1.0))])
        # clf = Pipeline([('cls', LinearSVC(random_state=42, C=1.0))])

        X_train_matrix = create_custom_features(X_train)

        t0 = time.time()
        clf.fit(X_train_matrix, Y_train)
        train_time = time.time() - t0
        print("Training time: ", train_time)

        t = time.time()
        Y_pred = clf.predict(X_dev)
        test_time = time.time() - t
        print("Testing time: ", test_time)

        acc = accuracy_score(Y_dev, Y_pred)
        # Calculating the cross validation scores
        # pred = cross_val_predict(clf, X_train_matrix, Y_train)
        # print("\nClassification report for cross-validation: \n {}".format(classification_report(Y_train, pred, digits=3)))

        print("\nFinal accuracy: {}".format(acc))
        print(get_confusion_matrix(Y_dev, Y_pred, clf))
        print("\nClassification report: \n {}".format(classification_report(Y_dev, Y_pred, target_names=clf.classes_,
                                                                            digits=3)))
    else:
        if args.tfidf:
            # vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
            # vec = TfidfVectorizer(tokenizer=preprocess)
            # vec = TfidfVectorizer(tokenizer=spacy_pos)
            vec = TfidfVectorizer(tokenizer=spacy_pos, ngram_range=(1, 3), use_idf=False, norm="l2")
        else:
            # Bag of Words vectorizer

            # vec = CountVectorizer(preprocessor=identity, tokenizer=identity)
            # vec = CountVectorizer(tokenizer=spacy_pos, ngram_range=(1, 3))
            vec = CountVectorizer(tokenizer=spacy_pos, ngram_range=(1, 3), min_df=3)
            # vec = CountVectorizer(tokenizer=spacy_pos, ngram_range=(1, 3), min_df=0.01)
            # vec = CountVectorizer(tokenizer=spacy_pos, ngram_range=(1, 3), max_df=0.95)
            # vec = CountVectorizer(tokenizer=spacy_pos, ngram_range=(1, 3), max_features=5000)

            # n1 = CountVectorizer(tokenizer=spacy_pos, ngram_range=(1, 1), min_df=2)
            # n2 = CountVectorizer(tokenizer=spacy_pos, ngram_range=(2, 2), min_df=3)
            # n3 = CountVectorizer(tokenizer=spacy_pos, ngram_range=(3, 3), min_df=5)

            # union = FeatureUnion([("n1", n1), ("n2", n2), ("n3", n3)])

        print("-----SVC with kernel=linear-----")
        svm_svc(vec, X_train, Y_train, X_dev, Y_dev)
        print("-----SVC with kernel=rbf-----")
        svm_svc(vec, X_train, Y_train, X_dev, Y_dev, kernel="rbf")
        print("-----LinearSVC-----")
        svm_linear_svc(vec, X_train, Y_train, X_test, Y_test)
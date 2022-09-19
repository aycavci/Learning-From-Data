#!/usr/bin/env python

""" This Pyhton script trains Naive Bayes, Decision Tree, Random Forest and K-nearest Neighbors
    classifier for multi-class classification on text data set, and prints some predictive analytics. """

import sys
import argparse
import random
import pandas as pd
import numpy as np
import time
import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import graphviz


def create_arg_parser():
    parser = argparse.ArgumentParser()
    """ Below argument is used for parameter tuning purposes (to generate train, dev and test set from reviews.txt) """
    # parser.add_argument("-i", "--input_file", default='reviews.txt', type=str,
    #                     help="Training set file to learn from (default reviews.txt)")
    parser.add_argument("-i", "--trainset", default='trainset.txt', type=str,
                        help="Training set file to learn from (default trainset.txt)")
    parser.add_argument("-ts", "--testset", default='testset.txt', type=str,
                        help="Test set file to learn from (default testset.txt)")
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    """ Below argument is used for split_data function (to generate train, dev and test set from reviews.txt) """
    # parser.add_argument("-tp", "--test_percentage", default=0.15, type=float,
    #                     help="Percentage of the data that is used for the test set (default 0.15)")
    # """ Below argument is used for split_data function (to generate train, dev and test set from reviews.txt) """
    # parser.add_argument("-sh", "--shuffle", action="store_true",
    #                     help="Shuffle data set before splitting in train/test")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    """ This function reads the txt file, and seperates labels and text IDs from each text item/line.
    It returns text/tweet, its label (dvd, music etc., and text IDs (530.txt etc.)) """
    documents = []
    labels = []
    ids = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            # 6-class problem: books, camera, dvd, health, music, software
            ids.append(tokens[2])
            labels.append(tokens[0])
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


def naive_bayes(vec, X_train, Y_train, X_test, Y_test):
    # Combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])

    """ Below code used for experimenting FeatureUnion package combining two vectorizers. """

    # TF-IDF vectorizer
    # vec1 = TfidfVectorizer(preprocessor=identity, tokenizer=identity)

    # Bag of Words vectorizer
    # vec2 = CountVectorizer(preprocessor=identity, tokenizer=identity)

    # combined_features = FeatureUnion([("vec1", vec1), ("vec2", vec2)])
    # classifier = Pipeline([('vec', combined_features), ('cls', MultinomialNB())])

    # Fits the training dataset as features (data) and labels (target) into the Naive Bayes' model.
    t0 = time.time()
    classifier.fit(X_train, Y_train)
    train_time = time.time() - t0
    print("Training time: ", train_time)

    # Given a trained model, predict the label of a new set of data.

    t = time.time()
    Y_pred = classifier.predict(X_test)
    test_time = time.time() - t
    print("Testing time: ", test_time)

    # Calculates the accuracy score of the trained model by comparing predicted labels with actual labels.
    acc = accuracy_score(Y_test, Y_pred)
    # Calculating the cross validation scores
    scores = cross_val_score(classifier, X_train, Y_train, scoring='accuracy', cv=5)

    print("\nFinal accuracy: {}".format(acc))
    # Calculating the average of five results from 5-fold
    print("\nAverage of 5-fold scores: ", scores.mean())
    print(get_confusion_matrix(Y_test, Y_pred, classifier))
    print("\nClassification report: \n {}".format(
        classification_report(Y_test, Y_pred, target_names=classifier.classes_)))


def decision_tree_classifier(vec, X_train, Y_train, X_test, Y_test):
    clf = Pipeline([('vec', vec), ('cls', DecisionTreeClassifier(max_depth=100, min_samples_leaf=2,
                                                                 min_samples_split=50,
                                                                 random_state=42))])
    t0 = time.time()
    clf = clf.fit(X_train, Y_train)
    train_time = time.time() - t0
    print("Training time: ", train_time)

    t = time.time()
    Y_pred = clf.predict(X_test)
    test_time = time.time() - t
    print("Testing time: ", test_time)

    acc = accuracy_score(Y_test, Y_pred)
    scores = cross_val_score(clf, X_train, Y_train, scoring='accuracy', cv=5)

    print("\nFinal accuracy: {}".format(acc))
    print("\nAverage of 5-fold scores: ", scores.mean())
    print(get_confusion_matrix(Y_test, Y_pred, clf))
    print("\nClassification report: \n {}".format(
        classification_report(Y_test, Y_pred, target_names=clf.classes_)))


def random_forest_classifier(vec, X_train, Y_train, X_test, Y_test):
    clf = Pipeline([('vec', vec), ('cls', RandomForestClassifier(n_estimators=300, max_depth=100, min_samples_leaf=2,
                                                                 min_samples_split=10,
                                                                 random_state=42))])
    t0 = time.time()
    clf = clf.fit(X_train, Y_train)
    train_time = time.time() - t0
    print("Training time: ", train_time)

    t = time.time()
    Y_pred = clf.predict(X_test)
    test_time = time.time() - t
    print("Testing time: ", test_time)

    acc = accuracy_score(Y_test, Y_pred)
    scores = cross_val_score(clf, X_train, Y_train, scoring='accuracy', cv=5)

    print("\nFinal accuracy: {}".format(acc))
    print("\nAverage of 5-fold scores: ", scores.mean())
    print(get_confusion_matrix(Y_test, Y_pred, clf))
    print("\nClassification report: \n {}".format(
        classification_report(Y_test, Y_pred, target_names=clf.classes_)))


def knn_classifier(vec, X_train, Y_train, X_test, Y_test, neighbors):
    clf = Pipeline([('vec', vec), ('cls', KNeighborsClassifier(n_neighbors=neighbors, weights='distance'))])

    t0 = time.time()
    clf = clf.fit(X_train, Y_train)
    train_time = time.time() - t0
    print("Training time: ", train_time)

    t = time.time()
    Y_pred = clf.predict(X_test)
    test_time = time.time() - t
    print("Testing time: ", test_time)

    acc = accuracy_score(Y_test, Y_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(Y_test, Y_pred, average=None,
                                                                          labels=clf.classes_)
    labels = clf.classes_

    x_axis_val = []
    y_axis_val = []

    x_axis_val.append("accuracy")
    y_axis_val.append(acc)

    for label in labels:
        x_axis_val.append(label)
    for score in f_score:
        y_axis_val.append(score)

    scores = cross_val_score(clf, X_train, Y_train, scoring='accuracy', cv=5)

    print("\nFinal accuracy: {}".format(acc))
    print("\nAverage of 5-fold scores: ", scores.mean())
    print(get_confusion_matrix(Y_test, Y_pred, clf))
    print("\nClassification report: \n {}".format(
        classification_report(Y_test, Y_pred, target_names=clf.classes_)))

    return x_axis_val, y_axis_val


if __name__ == "__main__":
    args = create_arg_parser()

    """ Below code refactored for the format python LFD_assignment2.py -i <trainset> -ts <testset>.
        Normally, it is used with split_data function to experiment with different classifiers. """

    # X_full, Y_full, ids = read_corpus(args.input_file)

    # Reads the text file and and return texts and their labels by tokenizing them, for both training
    # and test data set.

    X_train, Y_train, ids = read_corpus(args.trainset)
    X_test, Y_test, ids = read_corpus(args.testset)

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    else:
        # Bag of Words vectorizer
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity)

    """ Below code used to generate development set to tune model parameters. """

    # X_train, Y_train, ids_train, X_dev, Y_dev, ids_dev, X_test, Y_test, ids_test = split_data(X_full, Y_full, ids,
    #                                                                                           args.test_percentage,
    #                                                                                           args.shuffle)

    """ Below code used for splitting data to test the trained models after tuning the parameters."""
    # X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.20, random_state=42)

    print("-----BEST MODEL: Random Forest Classifier-----")
    random_forest_classifier(vec, X_train, Y_train, X_test, Y_test)
    print("-----Naive Bayes Classifier-----")
    naive_bayes(vec, X_train, Y_train, X_test, Y_test)
    print("-----Decision Tree Classifier-----")
    decision_tree_classifier(vec, X_train, Y_train, X_test, Y_test)
    print("-----K-Nearest Neighbors Classifier combined with TF-IDF Vectorizer-----")
    x_axis_val, y_axis_val = knn_classifier(TfidfVectorizer(preprocessor=identity, tokenizer=identity), X_train,
                                            Y_train, X_test, Y_test, 8)

    """ Below code only runs with split_data function which is commented out above,
        since parameters are tested by using development set. """
    # k_neighbors = [3, 5, 8, 10]
    #
    # for k in k_neighbors:
    #     x_axis_val, y_axis_val = knn_classifier(vec, X_train, Y_train, X_dev, Y_dev, k)
    #     plt.plot(x_axis_val, y_axis_val, label="k=" + str(k))
    #
    # plt.xlabel("Predictive analytics")
    # plt.ylabel("Scores")
    # plt.title("Model accuracy and class f1-scores of K-nearest Neighbors classifier for different K values using "
    #           "TF-IDF Vectorizer")
    # plt.legend()
    # plt.show()

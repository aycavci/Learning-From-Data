#!/usr/bin/env python
import argparse
import random
import pandas as pd
import numpy as np
import time
import sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, cross_val_predict

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC, LinearSVC, SVR
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
    
    parser.add_argument("-i", "--trainset", default='trainset.txt', type=str,
                        help="Training set file to learn from (default trainset.txt)")
    parser.add_argument("-ts", "--testset", default='testset.txt', type=str,
                        help="Test set file to learn from (default testset.txt)")
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
	# -cr flag to run the algorithm with custom features
    parser.add_argument("-cr", "--create_custom_features", action="store_true",
                        help="Create custom feature matrix and train the svm model")
    
    args = parser.parse_args()
    return args



# Removing Stopwords, Punctuation and Applying Lemmatization
def preprocess(x):
    doc = nlp(x)
    processed = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return processed

# Adding Pos Tag with the corresponding words
def spacy_pos(x):
    doc = nlp(x)
    cleaned_doc = [token.lemma_ + '_' + token.pos_ for token in doc if not token.is_stop and not token.is_punct]
    return cleaned_doc


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





def identity(x):
    """ Dummy function that just returns the input. """
    return x


def get_confusion_matrix(Y_test, Y_pred, clf):
    """ Get the confusion matrix. """

    result = confusion_matrix(Y_test, Y_pred, labels=clf.classes_)
    result_panda = pd.DataFrame(list(result), columns=clf.classes_, index=clf.classes_)

    print("Confusion Matrix: \n")
    return result_panda

# Calculating number of Adjectives in a text
def count_adj(txt):
    return sum([1 for token in nlp(txt) if token.pos_ == 'ADJ'])

# Claculating Named Entity in a Text
def count_named_entity(txt):
    doc = nlp(txt)
    return len(doc.ents)

# Calculating Number of Sentence
def count_sentence(txt):
    doc = nlp(txt)
    assert doc.has_annotation("SENT_START")
    return len([sent.text for sent in doc.sents])


def custom_features(txt):
    dic = dict()
    
    dic['count_sentence'] = count_sentence(txt)
    
    """Number of words count in a text"""
    dic['count_words'] = len(nlp(txt))
    
    dic['adj_count'] = count_adj(txt)
    
    dic['entity_count'] = count_named_entity(txt)
    
    
    return dic


def svm_svc(vec, X_train, Y_train, X_test, Y_test, kernel="linear", C=1.0):
    if kernel == "rbf":
        # Combine the vectorizer with a SVC classifier
        clf = Pipeline([('vec', vec), ('cls', SVC(kernel="rbf", C=C, gamma=.15))])
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

def svm_svr(vec, X_train, Y_train, X_test, Y_test, C=1.0):
    labels = []
    for item in Y_train:
        if item == 'neg':
            labels.append(0)
        else:
            labels.append(1)        
            
    clf = Pipeline([('vec', vec), ('cls', SVR(kernel='linear'))])

    t0 = time.time()
    clf.fit(X_train, labels)
    train_time = time.time() - t0
    print("Training time: ", train_time)

    t = time.time()
    Y_pred = clf.predict(X_test)
    
    new_pred = []
    for value in Y_pred:
        if value >= 0.5:
            new_pred.append('pos')
        else:
            new_pred.append('neg')
    
    test_time = time.time() - t
    print("Testing time: ", test_time)

    acc = accuracy_score(Y_test, new_pred)

    print("\nFinal accuracy: {}".format(acc))
    #print(get_confusion_matrix(Y_test, Y_pred, clf))
    print("\nClassification report: \n {}".format(classification_report(Y_test, new_pred, target_names=['neg','pos'])))

def best_svm(X_train, Y_train, X_test, Y_test, kernel="rbf", C=1.0):
    vec = TfidfVectorizer(tokenizer=spacy_pos)
    clf = Pipeline([('vec', vec), ('cls', SVC(kernel=kernel, C=C))])
    t0 = time.time()
    clf.fit(X_train, Y_train)
    train_time = time.time() - t0
    print("Training time: ", train_time)

    # Given a trained model, predict the label of a new set of data.

    t = time.time()
    Y_pred = clf.predict(X_test)
    print(Y_pred[20:50])
    test_time = time.time() - t
    print("Testing time: ", test_time)
    
    # Calculates the accuracy score of the trained model by comparing predicted labels with actual labels.
    acc = accuracy_score(Y_test, Y_pred)
    print("\nFinal accuracy: {}".format(acc))
    print(get_confusion_matrix(Y_test, Y_pred, clf))
    print("\nClassification report: \n {}".format(classification_report(Y_test, Y_pred, target_names=clf.classes_,
                                                                        digits=3)))
    
    # Calculating the cross validation scores
    pred = cross_val_predict(clf, X_train, Y_train, cv=5)
    print("\nClassification report for cross-validation: \n {}".format(classification_report(Y_train, pred, digits=3)))


    
    


if __name__ == "__main__":
    args = create_arg_parser()

    """ Below code refactored for the format python LFD_assignment2.py -i <trainset> -ts <testset>.
        Normally, it is used with split_data function to experiment with different classifiers. """

    # X_full, Y_full, ids = read_corpus(args.input_file)
	
	# Reads the text file and and return texts and their labels by tokenizing them, for both training
    # and test data set.

    X_train, Y_train, ids = read_corpus(args.trainset)
    X_test, Y_test, ids = read_corpus(args.testset)

    """ Below code used to generate development set to tune model parameters. """

    # X_train, Y_train, ids_train, X_dev, Y_dev, ids_dev, X_test, Y_test, ids_test = split_data(X_full, Y_full, ids,
    #                                                                                           args.test_percentage,
    #                                                                                           args.shuffle)

    """ Below code used for splitting data to test the trained models after tuning the parameters."""
    # X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.20, random_state=42)


    if args.create_custom_features:
        print("-----SVC with custom features-----")
        
        #creating custom features dictionary
        X_train_dic =  [custom_features(txt) for txt in X_train]
        
        #Applying Bag-of-words on text
        word_matr = CountVectorizer().fit_transform(X_train)
        #Applying Vetcorizer on custom data
        dic_matr = DictVectorizer().fit_transform(X_train_dic)
        
        
        X_train_matr = sp.hstack((word_matr, dic_matr), format='csr')
        
        clf = SVC(kernel="linear")
        pred = cross_val_predict(clf, X_train_matr, Y_train)
        
        
        # Calculating the cross validation scores
        
        print("\nClassification report for cross-validation: \n {}".format(classification_report(Y_train, pred, digits=3)))

        
    else:
		# Convert the texts to vectors
        if args.tfidf:
            vec = TfidfVectorizer()
            # vec = TfidfVectorizer(tokenizer=preprocess)
            # vec = TfidfVectorizer(tokenizer=spacy_pos)
            #vec = TfidfVectorizer(tokenizer=spacy_pos, ngram_range=(1, 3), use_idf=False, norm="l2")
        else:
            """ Bag of Words vectorizer """
            
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

        #print("-----SVC with kernel=linear-----")
        #svm_svc(vec, X_train, Y_train, X_test, Y_test)
        #print("-----SVC with kernel=rbf-----")
        #svm_svc(vec, X_train, Y_train, X_test, Y_test, kernel="rbf")
        #print("-----LinearSVC-----")
        #svm_linear_svc(vec, X_train, Y_train, X_test, Y_test)
        #print("-----SVR with kernel=linear-----")
        #svm_svr(vec, X_train, Y_train, X_test, Y_test)
        print("-----BEST MODEL-----")
        best_svm(X_train, Y_train, X_test, Y_test, C=1.0)

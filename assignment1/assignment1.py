#!/usr/bin/env python

'''TODO: This Pyhton script trains a Naive Bayes classifier for binary and multi-class classification on text data set and print some predicitive analytics'''

import sys
import argparse
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import pickle


def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_file", default='reviews.txt', type=str,
						help="Input file to learn from (default reviews.txt)")
	parser.add_argument("-s", "--sentiment", action="store_true",
						help="Do sentiment analysis (2-class problem)")
	parser.add_argument("-t", "--tfidf", action="store_true",
						help="Use the TF-IDF vectorizer instead of CountVectorizer")
	parser.add_argument("-tp", "--test_percentage", default=0.20, type=float,
						help="Percentage of the data that is used for the test set (default 0.20)")
	parser.add_argument("-sh", "--shuffle", action="store_true",
						help="Shuffle data set before splitting in train/test")
	args = parser.parse_args()
	return args


def read_corpus(corpus_file, use_sentiment):
	'''TODO: This function reads the reviews.txt file, and seperates labels and text IDs from each text item/line.
	It returns text/tweet, its label (neg, pos, dvd, music etc., and text IDs (530.txt etc.))'''
	documents = []
	labels = []
	ids = []
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			tokens = line.strip().split()
			documents.append(tokens[3:])
			if use_sentiment:
				# 2-class problem: positive vs negative
				ids.append(tokens[2])
				labels.append(tokens[1])
			else:
				# 6-class problem: books, camera, dvd, health, music, software
				ids.append(tokens[2])
				labels.append(tokens[0])
	return documents, labels, ids


def shuffle_dependent_lists(l1, l2, l3):
	'''Shuffle two lists, but keep the dependency between them'''
	tmp = list(zip(l1, l2, l3))
	# Seed the random generator so results are consistent between runs
	random.Random(123).shuffle(tmp)
	return zip(*tmp)


def split_data(X_full, Y_full, ids, test_percentage, shuffle):
	'''TODO: Splits the data set as training (80% from beginning of the file) and test set (20% after split_point).
	It returns training and test set texts, and their labels, and their IDs.'''
	split_point = int((1.0 - test_percentage)*len(X_full))
	# TODO: It shuffles the data set before splitting the data if shuffle returns True.
	if shuffle:
		X_full, Y_full, ids = shuffle_dependent_lists(X_full, Y_full, ids)
	X_train = X_full[:split_point]
	Y_train = Y_full[:split_point]
	ids_train = ids[:split_point]
	X_test = X_full[split_point:]
	Y_test = Y_full[split_point:]
	ids_test = ids[split_point:]
	return X_train, Y_train, ids_train, X_test, Y_test, ids_test


def identity(x):
	'''Dummy function that just returns the input'''
	return x

def get_confusion_matrix(Y_test, Y_pred):
	'''Get the confusion matrix'''
	if args.sentiment:
		result = confusion_matrix(Y_test, Y_pred, labels=['pos', 'neg'])
		result_panda = pd.DataFrame(list(result), columns=['pos', 'neg'], index=['pos', 'neg'])
	else:
		result = confusion_matrix(Y_test, Y_pred, labels=['health', 'dvd', 'software', 'music', 'books', 'camera'])
		result_panda = pd.DataFrame(list(result), columns=['health', 'dvd', 'software', 'music', 'books', 'camera'], index=['health', 'dvd', 'software', 'music', 'books', 'camera'])
	
	print("Confusion Matrix:\n")
	return result_panda

def get_metrics(Y_test, Y_pred):
	'''Get precision, recall, fscore and support for each class'''
	if args.sentiment:
		result = list(precision_recall_fscore_support(Y_test, Y_pred, average=None, labels=['pos', 'neg']))
		result_panda = pd.DataFrame(list(result), columns=['pos', 'neg'], index=['precision', 'recall', 'fscore', 'support'])
	else:
		result = list(precision_recall_fscore_support(Y_test, Y_pred, average=None, labels=['health', 'dvd', 'software', 'music', 'books', 'camera']))
		result_panda = pd.DataFrame(list(result), columns=['health', 'dvd', 'software', 'music', 'books', 'camera'], index=['precision', 'recall', 'fscore', 'support'])
	result_panda = result_panda.transpose()
	result_panda['support'] = result_panda['support'].astype(int)
	print("\nPrecision, Recall, F-score and Support: \n")
	return result_panda

def get_posterior_prob(X_test, Y_test, ids_test, clf):
	print("Classes: ", clf.classes_)
	result = clf.predict_proba(X_test)
	result_panda = pd.DataFrame(result, columns=clf.classes_, index=[x for x in ids_test])
	with open('posterior.txt', 'w') as f:
		for item in result:
			if item[0] > item[1]:
				f.write("%s\n" % "neg")
			else:
				f.write("%s\n" % "pos")
	with open('labels.txt', 'w') as f:
		for item in Y_test:
			f.write("%s\n" % item)
	print("\nPosterior probabilities of each class for given data instance: \n")
	return result_panda

def get_prior_prob(Y_test):
	df = pd.DataFrame(Y_test, columns=['classes'], index=[x for x in range(0, len(Y_test))])
	print("\nPrior probabilities of each class: \n")
	print(df['classes'].value_counts()/len(Y_test))
	
if __name__ == "__main__":
	args = create_arg_parser()

	# TODO: Reads the text file and and return texts and their lables by tokenizing them, then splits the data for training and testing purposes. 
	X_full, Y_full, ids = read_corpus(args.input_file, args.sentiment)
	X_train, Y_train, ids_train, X_test, Y_test, ids_test = split_data(X_full, Y_full, ids, args.test_percentage, args.shuffle)

	# Convert the texts to vectors
	# We use a dummy function as tokenizer and preprocessor,
	# since the texts are already preprocessed and tokenized.
	if args.tfidf:
		vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
	else:
		# Bag of Words vectorizer
		vec = CountVectorizer(preprocessor=identity, tokenizer=identity)

	# Combine the vectorizer with a Naive Bayes classifier
	classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])

	# TODO: Fits the training dataset as features (data) and labels (target) into the Naive Bayes' model.
	classifier.fit(X_train, Y_train)
	
	filename = "./model/naive_bayes.pkl"  

	with open(filename, 'wb') as file:	
		pickle.dump(classifier, file)

	# TODO: Given a trained model, predict the label of a new set of data.
	Y_pred = classifier.predict(X_test)

	# TODO: Calculates the accuracy score of the trained model by comparing predicted labels with actual labels.
	acc = accuracy_score(Y_test, Y_pred)

	print(get_confusion_matrix(Y_test, Y_pred))
	print(get_metrics(Y_test, Y_pred))
	print("\nFinal accuracy: {}".format(acc))
	# Prints posterior probabilities of each class for each text item in test set
	print(get_posterior_prob(X_test, Y_test, ids_test, classifier))
	# Prints prior probabilities of each class in test set
	print(get_prior_prob(Y_test))

	# Cross Validation

	# Calculating the scores 
	scores = cross_val_score(classifier, X_train, Y_train, scoring='accuracy', cv=5)

	# Calculating the average of five results from 5-fold
	print("\nAverage of 5-fold scores: ", scores.mean())


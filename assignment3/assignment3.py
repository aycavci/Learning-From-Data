import sys
import argparse
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict

from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


import time
from tqdm import tqdm

import spacy

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
string.punctuation 
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nlp = spacy.load("en_core_web_sm")

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='trainset.txt', type=str,
                        help="Input file to learn from (default trainset.txt)")
    
    parser.add_argument("-ts", "--test_file", default='testset.txt', type=str,
                        help="Test file to test the model (default testset.txt)")
    
    
    args = parser.parse_args()
    return args

# Removing Punctuation, Stop words and applying Porter Stemmer
ps=PorterStemmer()
def preprocess(x):
    doc = nlp(x)
    processed=[token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    #print(processed)
    return processed


def spacy_pos(x):
    doc = nlp(x)
    cleaned_doc = [token.lemma_+ '_' + token.pos_ for token in doc if not token.is_stop and not token.is_punct]
    return cleaned_doc
    
        
    





def read_corpus(corpus_file):
    '''TODO: This function reads the reviews.txt file, and seperates labels and text IDs from each text item/line.
    It also convert each word into lower case, removes non-ASCII character and more spaces.
    Finally, It returns text/tweet, its label (neg, pos, dvd, music etc., and text IDs (530.txt etc.))'''
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






def identity(x):
    '''Dummy function that just returns the input'''
    return x




if __name__ == "__main__":
    args = create_arg_parser()
    
    
    # TODO: Reads the text file and and return texts and their lables by tokenizing them. 
    X_train, Y_train, train_ids = read_corpus(args.train_file)
    X_test, Y_test, test_ids  = read_corpus(args.test_file)
    

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    
    
    #union = FeatureUnion([("count", vec), ("pos", pos)])
    vec1 = TfidfVectorizer(tokenizer=preprocess)
    vec2 = TfidfVectorizer(tokenizer=spacy_pos)
    vec3 = TfidfVectorizer(tokenizer=spacy_pos, ngram_range=(1,3), use_idf=False)
    vec4 = CountVectorizer()
    vec5 = CountVectorizer(tokenizer=spacy_pos, ngram_range=(1,3))
    
    n1    = CountVectorizer(tokenizer=spacy_pos, ngram_range=(1,1), min_df=2)
    n2    = CountVectorizer(tokenizer=spacy_pos, ngram_range=(2,2), min_df=3)
    n3    = CountVectorizer(tokenizer=spacy_pos, ngram_range=(3,3), min_df=5)
		
    
    #Applying Pipeline for doing multiple function simultaneously 
    classifier = Pipeline([('vec', vec2), ('cls', LinearSVC())])
    #Calculating Training Time
    t0 = time.time()
    classifier.fit(X_train, Y_train)
    train_time = time.time() - t0
    #print("training time: {:.4f}".format(train_time))
    
    #Calculating Testing Time
    t0 = time.time()
    Y_pred = classifier.predict(X_test)
    test_time = time.time() - t0
    #print("testing time: {:.4f}".format(test_time))
    # TODO: Calculates the f1 score of the trained model by comparing predicted labels with actual labels.
    acc = accuracy_score(Y_test, Y_pred)
    
    #It shows Evaluation report based on Recall, Precesion, f1_score and Support
    print(classification_report(Y_test, Y_pred, target_names=['neg', 'pos']))
    
    print("\nFinal Accuracy: {:.4f}".format(acc))
    
        
        
        
    
    # Cross Validation

    # Calculating the scores 
    #scores = cross_val_score(classifier, X_train, Y_train, scoring='accuracy', cv=5)

    # Calculating the average of five results from 5-fold Cross Validation
    #print("\nAverage of 5-fold CV score: {:.4f}".format(scores.mean()))
    
    #It shows a model whether overfit or underfit
    #print('Training set score: {:.4f}'.format(classifier.score(X_train, Y_train)))

    #print('Test set score: {:.4f}'.format(classifier.score(X_test, Y_test)))
    
        
    


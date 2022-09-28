import json
import argparse

corpus_file = "test.txt"

def read_corpus(corpus_file):
    '''Read in review data set and returns docs and labels'''
    documents = []
    labels = []
    count = 0
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[3:]).strip())
            # 6-class problem: books, camera, dvd, health, music, software
            labels.append(tokens.split()[0])
    for label in labels:
      if label == 'camera':
        count+=1
    print(count)
    
read_corpus(corpus_file)
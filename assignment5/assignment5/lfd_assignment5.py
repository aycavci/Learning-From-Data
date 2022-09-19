#!/usr/bin/env python

'''TODO: add high-level description of this Python script'''

import random as python_random
import json
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Embedding, LSTM
from keras.initializers import Constant
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='train.txt', type=str,
                        help="Input file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev.txt',
                        help="Separate dev set to read in (default dev.txt)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-e", "--embeddings", default='glove_reviews.json', type=str,
                        help="Embedding file we are using (default glove_reviews.json)")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    '''Read in review data set and returns docs and labels'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[3:]).strip())
            # 6-class problem: books, camera, dvd, health, music, software
            labels.append(tokens.split()[0])
    return documents, labels


def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def create_model(Y_train, emb_matrix):
    '''Create the Keras model to use'''
    # Define settings, you might want to create cmd line args for them
    learning_rate = 0.01
    loss_function = 'categorical_crossentropy'
    optim = SGD(learning_rate=learning_rate)
    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(Y_train[0])
    # Now build the model
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix),trainable=False))
    # Here you should add LSTM layers (and potentially dropout)
    raise NotImplementedError("Add LSTM layer(s) here")
    # Ultimately, end with dense layer with softmax
    model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="softmax"))
    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev):
    '''Train the model here. Note the different settings you can experiment with!'''
    # Potentially change these to cmd line args again
    # And yes, don't be afraid to experiment!
    verbose = 1
    batch_size = 16
    epochs = 50
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # Finally fit the model to our data
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback], batch_size=batch_size, validation_data=(X_dev, Y_dev))
    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, X_dev, Y_dev, "dev")
    return model


def test_set_predict(model, X_test, Y_test, ident):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)
    print('Accuracy on own {1} set: {0}'.format(round(accuracy_score(Y_test, Y_pred), 3), ident))


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    embeddings = read_embeddings(args.embeddings)

    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # Create model
    model = create_model(Y_train, emb_matrix)

    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # Train the model
    model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin)

    # Do predictions on specified test set
    if args.test_file:
        # Read in test set and vectorize
        X_test, Y_test = read_corpus(args.test_file)
        Y_test_bin = encoder.fit_transform(Y_test)
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        # Finally do the predictions
        test_set_predict(model, X_test_vect, Y_test_bin, "test")

if __name__ == '__main__':
    main()

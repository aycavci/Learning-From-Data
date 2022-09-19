#!/usr/bin/env python

import random as python_random
import json
import argparse
import numpy
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
# Make reproducible as much as possible
numpy.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default='train_NE.txt', type=str,
                        help="Input file to learn from (default train_NE.txt)")
    parser.add_argument("-e", "--embeddings", default='glove_filtered.json', type=str,
                        help="Embedding file we are using (default glove_filtered.json)")
    parser.add_argument("-tp", "--test_percentage", default=0.20, type=float,
                        help="Percentage of the data that is used for the test set (default 0.20)")
    parser.add_argument("-ts", "--test_set", type=str,
                        help="Separate test set to read from, instead of data splitting (or both)")
    parser.add_argument("-o", "--output_file", type=str,
                        help="Output file to which we write predictions for test set")
    args = parser.parse_args()
    if args.test_set and not args.output_file:
        raise ValueError("Always specify an output file if you specify a separate test set!")
    if args.output_file and not args.test_set:
        raise ValueError("Output file is specified but test set is not -- probably you made a mistake")
    return args


def write_to_file(lst, out_file):
    '''Write list to file'''
    with open(out_file, "w") as out_f:
        for line in lst:
            out_f.write(line.strip() + '\n')
    out_f.close()


def read_corpus(corpus_file):
    '''Read in the named entity data from a file'''
    names = []
    labels = []
    for line in open(corpus_file, 'r'):
        name, label = line.strip().split()
        names.append(name)
        labels.append(label)
    return names, labels


def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: numpy.array(embeddings[word]) for word in embeddings}


def vectorizer(words, embeddings):
    '''Turn words into embeddings, i.e. replace words by their corresponding embeddings'''
    return numpy.array([embeddings[word] for word in words])


def split_data(X_full, Y_full, test_percentage):
    '''Split data based on percentage we use for testing'''
    split_point = int((1.0 - test_percentage)*len(X_full))
    X_train = X_full[:split_point]
    Y_train = Y_full[:split_point]
    X_test = X_full[split_point:]
    Y_test = Y_full[split_point:]
    return X_train, Y_train, X_test, Y_test


def create_model(X_train, Y_train):
    '''Create the Keras model to use'''
    # Define settings, you might want to create cmd line args for them (or some other more reproducible method)
    learning_rate = 0.005
    # Linear layer for now, but you can experiment!
    activation = "softmax"
    # Start with MSE, but again, experiment!
    loss_function = 'categorical_crossentropy'
    # SGD optimizer
    sgd = SGD(lr=learning_rate)

    # Now build the model
    model = Sequential()
    # First dense layer has the number of features as input and the number of labels as total units
    model.add(Dense(input_dim=X_train.shape[1], units=Y_train.shape[1]))
    model.add(Activation(activation))
    # Potentially add your own layers here. Note that you have to change the dimensions of the prev layer
    # so that your final output layer has the correct number of nodes
    # You could also think about using Dropout!
    # ...

    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=sgd, metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_train):
    '''Train the model here. Note the different settings you can experiment with!'''
    # Potentially change these to cmd line args again
    # And yes, don't be afraid to experiment!
    verbose = 1
    epochs = 10
    batch_size = 32
    # 10 percent of the training data we use to keep track of our training process
    # Use it to prevent overfitting!
    validation_split = 0.1
    # Finally fit the model to our data
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return model


def split_test_set_predict(model, X_test, Y_test):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = numpy.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = numpy.argmax(Y_test, axis=1)
    print('Accuracy on own test set: {0}'.format(round(accuracy_score(Y_test, Y_pred), 3)))


def separate_test_set_predict(test_set, embeddings, encoder, model, output_file):
    '''Do prediction on a separate test set for which we do not have a gold standard.
       Write predictions to a file'''
    # Read and vectorize data
    test_emb = vectorizer([x.strip() for x in open(test_set, 'r')], embeddings)
    # Make predictions
    pred = model.predict(test_emb)
    # Convert to numerical labels and back to string labels
    test_pred = numpy.argmax(pred, axis=1)
    labels = [encoder.classes_[idx] for idx in test_pred]
    # Finally write predictions to file
    write_to_file(labels, output_file)


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Read in the data and embeddings
    X_full, Y_full = read_corpus(args.input_file)
    embeddings = read_embeddings(args.embeddings)

    # Transform words to embeddings
    X_emb = vectorizer(X_full, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_bin = encoder.fit_transform(Y_full)  # Use encoder.classes_ to find mapping back
    X_train, Y_train, X_test, Y_test = split_data(X_emb, Y_bin, args.test_percentage)

    # Create model
    model = create_model(X_train, Y_train)

    # Train the model
    model = train_model(model, X_train, Y_train)

    # You might not always want to test on your own test set!
    # Remove the dummy if statement to do so anyway
    if False:
        split_test_set_predict(model, X_test, Y_test)

    # If we specified a test set, there are no gold labels available
    # Do predictions and print them to a separate file
    if args.test_set:
        separate_test_set_predict(args.test_set, embeddings, encoder, model, args.output_file)


if __name__ == '__main__':
    main()

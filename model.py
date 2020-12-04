"""
Final Project
Will Suratt
CS 021

Model
=====

This program will train a model to predict when opening straddle positions
with options will be profitable. Tests model's accuracy against random
guess accuracy and guessing all zeroes or all ones (output is binary).
Also displays model's loss and accuracy over epoch.

*** This program takes about 15 minutes to fully run. ***

"""
# Imports
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import random
import os

# Constants
# Number of sets of random predictions to average
NUM_RANDOM = 5

def main():
    """

    Begins program execution: Reads in data to a DataFrame and then uses this
    data to train the model (split into 70% train, 30% test). The validate 
    function is then called to test the model on the 30% of the data not used
    to train the model. Then the learning_results function is called to graph
    model loss and accuracy throughout its training.

    """
    # Read in data
    cwd = os.getcwd()
    data_path = cwd + '/data/'
    data = pd.read_csv(data_path + 'processed_data.csv')
    data.head() 

    # Shuffle DataFrame
    data = data.sample(frac=1).reset_index(drop=True)

    # Split data into numpy arrays for training and testing
    X_train = data.iloc[:int(len(data.index) * 0.7),4:8].values 
    y_train = data.iloc[:int(len(data.index) * 0.7),-1].values
    X_test = data.iloc[int(len(data.index) * 0.7)+1:,4:8].values 
    y_test = data.iloc[int(len(data.index) * 0.7)+1:,-1].values

    # Create model
    model = Sequential()
    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit model
    history = model.fit(X_train, y_train, epochs=10000, batch_size=32)

    # Save model
    model_path = cwd + '/model/'
    model.save(model_path + "stock_model.h5")

    # Validate model
    validate(model, X_test, y_test)

    # Show learning results
    learning_results(history)


def validate(model, X_test, y_test):
    """

    Three parameter variables, model, X_test (array), and y_test (array). 
    Tests model accuracy of predicting when straddle positions will be profitable.
    Also tests random guess accuracy and guessing all zeroes and all ones accuracy
    to compare and displays results. No return.

    """
    # Get predictions
    predictions = model.predict_classes(X_test)

    total = 0
    correct = 0

    # Check if predictions were correct
    for i in range(y_test.size):
        if predictions[i] == y_test[i]:
            correct += 1
        total += 1

    # Get accuracy
    model_accuracy = (correct / total) * 100

    total = 0
    correct = 0

    # Check if predictions were correct
    for i in range(y_test.size):
        if 0 == y_test[i]:
            correct += 1
        total += 1

    # Get accuracy
    guess_0_accuracy = (correct / total) * 100

    total = 0
    correct = 0

    # Check if predictions were correct
    for i in range(y_test.size):
        if 1 == y_test[i]:
            correct += 1
        total += 1

    # Get accuracy
    guess_1_accuracy = (correct / total) * 100

    # Get sum for random predictions' accuracy
    total_random = 0

    # Get NUM_RANDOM batches of random predictions to average
    for i in range(NUM_RANDOM):
        # Check if predictions were correct
        for i in range(y_test.size):
            if random.randint(0, 1) == y_test[i]:
                correct += 1
            total += 1

        # Get accuracy
        guess_random_accuracy = (correct / total) * 100
        total_random += guess_random_accuracy

    # Get average random accuracy
    guess_random_accuracy_avg = total_random / NUM_RANDOM

    # Display results
    print('\nmodel accuracy:', format(model_accuracy, ",.2f"))
    print('guess 0 accuracy:', format(guess_0_accuracy, ",.2f"))
    print('guess 1 accuracy:', format(guess_1_accuracy, ",.2f"))
    print('guess random accuracy:', format(guess_random_accuracy_avg, ",.2f"))


def learning_results(history):
    """

    One parameter variable, history. Plots model's accuracy over epoch and
    loss over epoch. No return.

    """
    # Plot model accuracy
    plt.plot(history.history['accuracy'], color="red", label="acc")
    plt.legend(loc="upper right")
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.show()

    # Plot model loss
    plt.plot(history.history['loss'], color="red", label="loss")
    plt.legend(loc="upper right")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

# Run program
main()
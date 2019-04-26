import numpy as np
from Classifier import *
from DataManager import *
import sklearn.neural_network as sknn


class Perceptron(Classifier):

    # Initialisation of the required parameters
    def __init__(self):
        # Regularization factor
        self.alpha = 0.01
        # Learning rate
        self.learning_rate = 0.01
        # Constant to multiply the hidden layer size,
        # 100*self.layers_factor units
        self.layer_factor = 100
        # Perceptron model
        self.model = None

    def train(self, x_train, y_train):
        """
        Train the model with the x_train features array
        and the y_train one-hot labels
        Return : nothing
        """
        # Construct the model with all the hyperparameters
        nb_units = self.layer_factor
        self.model = sknn.MLPClassifier(hidden_layer_sizes=(nb_units, nb_units, nb_units, nb_units),
                                        alpha=self.alpha,
                                        learning_rate_init=self.learning_rate,
                                        max_iter=40, verbose=False, tol=0.01)
        # Train the model weights
        self.model.fit(x_train, y_train)

    def test(self, x_test):
        """
        Test the model with the x_test features array
        Return : an array of predicted labels,
        as one-hot vectors for each class
        """
        results = self.model.predict(x_test)
        softmax = np.exp(results)
        softmax = softmax / np.sum(softmax, axis=1, keepdims=True)
        return softmax

    def cross_validation(self, x_train, y_train):
        """
        Train/test the model with different hyper parameters values,
        and select the best ones
        Return : nothing
        """
        best_accuracy = 0.0
        best_alpha = 0.01
        best_learning_rate = 0.01
        best_layer_factor = 1

        splits_length = int(x_train.shape[0]/10)

        # Testing combinations of hyperparameters
        for layer_factor in [2, 4, 8, 16, 32]:
            for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
                for learning_rate in [0.0001, 0.001, 0.01, 0.1, 1]:
                    # Test the model with hyperparameters values
                    self.layer_factor = layer_factor
                    self.alpha = alpha
                    self.learning_rate = learning_rate

                    accuracy_mean = 0.0
                    for K in range(10):
                        # Split the data into 10 folds
                        # and use the #K for validation
                        inf = int(K*splits_length)
                        sup = int((K+1)*splits_length)
                        x_val_fold = x_train[inf:sup]
                        x_train_fold = np.concatenate((x_train[:inf],
                                                       x_train[sup:]))
                        y_val_fold = y_train[inf:sup]
                        y_train_fold = np.concatenate((y_train[:inf],
                                                       y_train[sup:]))
                        # Test the model with the value of k
                        # and output a distribution
                        self.train(x_train_fold, y_train_fold)
                        results = self.test(x_val_fold)
                        # Compute accuracy and compare
                        # with the best value found
                        accuracy = self.accuracy(results, y_val_fold)
                        # Compute accuracy and compare
                        # with the best value found
                        accuracy_mean += accuracy
                    accuracy_mean /= 10
                    print("layer_factor =", layer_factor, ", alpha =", alpha,
                          ", learning_rate =", learning_rate,
                          ", Accuracy = ", accuracy_mean)
                    if accuracy_mean > best_accuracy:
                        best_accuracy = accuracy_mean
                        best_layer_factor = layer_factor
                        best_alpha = alpha
                        best_learning_rate = learning_rate
        # Select the best value found
        self.layer_factor = best_layer_factor
        self.alpha = best_alpha
        self.learning_rate = best_learning_rate
        print("Best values : layer_factor =", self.layer_factor,
              ", alpha =", self.alpha, ", learning_rate=", self.learning_rate)
        self.train(x_train, y_train)

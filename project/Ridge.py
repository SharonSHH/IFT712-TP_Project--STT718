import numpy as np
from Classifier import *
from DataManager import *
import sklearn.linear_model as sklm


class Ridge(Classifier):

    # Initialisation of the required parameters
    def __init__(self):
        # Regularization factor, for L2 penalty
        self.alpha = 0.1

        # Ridge model
        self.model = None

    def train(self, x_train, y_train):
        """
        Train the model with the x_train features
        array and the y_train one-hot labels
        Return : nothing
        """
        self.model = sklm.Ridge(alpha=self.alpha)
        self.model.fit(x_train, y_train)

    def test(self, x_test):
        """
        Test the model with the x_test features array
        Return : an array of predicted labels
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

        splits_length = int(x_train.shape[0]/10)

        # Testing combinations of hyperparameters
        for alpha in [0.000000001, 0.00000001, 0.0000001,
                      0.000001, 0.000005, 0.00001, 0.00005,
                      0.0001, 0.0005, 0.001, 0.01, 0.1, 1]:
            # Test the model with hyperparameters values
            self.alpha = alpha

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
                # Compute accuracy and compare with the best value found
                accuracy_mean += accuracy
            accuracy_mean /= 10

            print("alpha =", alpha, ", Accuracy = ", accuracy_mean)
            if accuracy_mean > best_accuracy:
                best_accuracy = accuracy_mean
                best_alpha = alpha
        # Select the best value found
        self.alpha = best_alpha
        print("Best values : alpha =", self.alpha)
        self.train(x_train, y_train)

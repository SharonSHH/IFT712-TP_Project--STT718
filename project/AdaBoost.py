import numpy as np
from Classifier import *
from DataManager import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


class AdaBoost(Classifier):
    '''Use AdaBoostClassifier to adjust the model'''

    def __init__(self):
        """Initialisation of the required parameters"""
        self.estimator = 25
        self.l_rate = 0.01
        self.clf = DecisionTreeClassifier(criterion='gini',
                                          min_samples_split=2,
                                          max_depth=11,
                                          min_samples_leaf=1)
        self.model = None

    def train(self, x_train, y_train):
        """Train the model with the x_train features and
        the y_train integer encoding
            Return : nothing"""
        self.model = AdaBoostClassifier(base_estimator=self.clf,
                                        n_estimators=self.estimator,
                                        learning_rate=self.l_rate)
        self.model.fit(x_train, y_train)

    def test(self, x_test):
            """
            Test the model with the x_test features array
            Return : an array of predicted labels,_
            as integer encoding for each class
            """
            return self.model.predict(x_test)

    def cross_validation(self, x_train, y_train):
        """
        Train/test the model with different hyper parameters values,_
        and select the best ones
        Return : nothing
        """
        best_estimator = 0
        best_l_rate = 0
        best_accuracy = 0
        max_k = 10
        splits_length = int(x_train.shape[0] / 10)

        for estimator in [25, 50, 75, 100, 200]:
            for l_rate in [0.01, 0.1, 1]:
                self.estimator = estimator
                self.l_rate = l_rate
                accuracy_mean = 0.0
                for K in range(max_k):
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
                    accuracy_mean += accuracy
                accuracy_mean /= 10
                print("n_estimator =", estimator, ", learning_rate =",
                      l_rate, ", Validation Accuracy = ", accuracy_mean)
                if accuracy_mean > best_accuracy:
                    best_accuracy = accuracy_mean
                    best_estimator = estimator
                    best_l_rate = l_rate

        # Select the best value found
        self.estimator = best_estimator
        self.l_rate = best_l_rate
        print("Best values : n_estimator =", estimator,
              ", learning_rate =", l_rate,
              ", Validation Accuracy = ", best_accuracy)
        self.train(x_train, y_train)

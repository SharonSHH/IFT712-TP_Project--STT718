import numpy as np
from Classifier import *
from DataManager import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pdb


class RandomForest(Classifier):
    """
    Using RandomForestClassifier to train Random Forest model
    """
    def __init__(self):
        self.estimators = 70
        self.cri_par = 'gini'

    def train(self, x_train, y_train):
        """
        Train the model with the x_train features array and
        the y_train one-hot labels
        Return : nothing
        """
        # Construct the model with all the hyperparameters
        self.model = RandomForestClassifier(
                        n_estimators=self.estimators, criterion=self.cri_par)
        # Train the model weights
        self.model.fit(x_train, y_train)

    def test(self, x_test):
        """
        Test the model with the x_test features array
        Return : an array of predicted labels,
        as one-hot vectors for each class
        """
        return self.model.predict(x_test)

    def cross_validation(self, x_train, y_train):
        """
        Train/test the model with different hyper parameters values,
        and select the best ones
        Return : nothing
        """
        best_estimator = 0
        best_cri_par = 'gini'
        best_accuracy = 0
        max_k = 10
        splits_length = int(x_train.shape[0]/10)
        pdb.set_trace()
        for criterion in ['gini', 'entropy']:
            for estimator in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                self.estimators = estimator
                self.cri_par = criterion
                accuracy_mean = 0.0
                logloss_mean = 0.0
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
                    #accuracy = accuracy_score(y_val_fold, results)
                    #print("Accuracy score is ", accuracy)
                    #print ("\nAdaBoost  - Train Classification Report\n", classification_report(y_val_fold, results))
                    accuracy = self.accuracy(results, y_val_fold)
                    log_loss = self.log_loss(results, y_val_fold)
                    # Compute accuracy and compare with the best value found
                    accuracy_mean += accuracy
                    logloss_mean += log_loss
                accuracy_mean /= 10
                logloss_mean /= 10
                print("estimator =", estimator, ", criterion =", criterion,
                      ", Validation Accuracy =", accuracy_mean,
                      ",logloss =", logloss_mean)
                if accuracy_mean > best_accuracy:
                    best_accuracy = accuracy_mean
                    best_estimator = estimator
                    best_cri_par = criterion

        # Select the best value found
        self.estimators = best_estimator
        self.cri_par = best_cri_par
        print("Best values : estimators =", self.estimators,
              ", criterion =", self.cri_par,
              ", Validation Accuracy =", best_accuracy)
        self.train(x_train, y_train)


if __name__ == '__main__':
    sm = RandomForest()
    dm = DataManager()
    dm.load_CSV("leaf-classification/train.csv","leaf-classification/test.csv")	
    x = dm.x_train
    y = dm.y_train
    sm.cross_validation(x, y)
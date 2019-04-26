import numpy as np
from sklearn import svm
from Classifier import *
from DataManager import *
import pdb


class SVM(Classifier):

    # Initialisation of the required parameters
    def __init__(self):
        # SVM penalty parameter, to control the softness of the separation
        self.C = 1
        # SVM kernel, which can be linear, rbf, sigmoidal, poly
        self.kernel = "linear"

        # If kernel == rbf or poly or sigmoid, controls the kernel coefficient
        self.gamma = 0.1

        # If kernel == rbf or poly or sigmoid,
        # controls the independent term in the kernel function
        self.coef0 = 0

        # If kernel == poly, controls the polynom degree
        self.degree = 1

        # Build the SVM model with sklearn
        self.model = None

    def train(self, x_train, y_train):
        """
        Train the model with the x_train features array
        and the y_train one-hot labels
        Return : nothing
        """
        # Construct the model with all the hyperparameters,
        # even though some may not be used
        self.model = svm.SVC(kernel=self.kernel, C=self.C,
                             gamma=self.gamma, degree=self.degree,
                             coef0=self.coef0)
        # Train the model
        self.model.fit(x_train, y_train)

    def test(self, x_test):
        """
        Test the model with the x_test features array
        Return : an array of predicted labels, as label for each class
        """
        return self.model.predict(x_test)

    def cross_validation(self, x_train, y_train):
        """
        Train/test the model with different hyper parameters values,
        and select the best ones
        Return : nothing
        """
        best_accuracy = 10.0
        best_C = 0.00001
        best_kernel = "rbf"
        # In the case of poly, rbf, sigmoid
        best_gamma = 0.001
        best_coef0 = 0
        # In the case of poly
        best_degree = 1
        splits_length = int(x_train.shape[0]/5)
        pdb.set_trace()

        # Testing combinations of hyperparameters
        for kernel in ["linear", "rbf", "poly", "sigmoid"]:
            # We only want to search gamma and degree for specific kernels
            # Default values, not used
            gamma_range = [0.1]
            degree_range = [1]
            coef0_range = [0]
            if kernel == "rbf" or kernel == "sigmoid":
                gamma_range = [0.0001, 0.001, 0.01, 0.1, 1]
                degree_range = [1]
                coef0_range = [-4, -2, 0, 2, 4]
            elif kernel == "poly":
                gamma_range = [0.0001, 0.001, 0.01, 0.1, 1]
                degree_range = [1, 2, 4, 8]
                coef0_range = [-4, -2, 0, 2, 4]
            # Find hyperparameters only if necessary
            for gamma in gamma_range:
                for degree in degree_range:
                    for coef0 in coef0_range:
                        for C in [0.1, 1, 10, 100, 1000, 10000]:
                            # Test the model with hyperparameters values
                            self.C = C
                            self.kernel = kernel
                            self.gamma = gamma
                            self.degree = degree
                            self.coef0 = coef0

                            accuracy_mean = 0.0
                            log_loss_mean = 0.0
                            for K in range(5):
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
                                log_loss = self.log_loss(results, y_val_fold)
                                accuracy_mean += accuracy
                                log_loss_mean += log_loss
                            accuracy_mean /= 5
                            log_loss_mean /= 5
                            print("kernel =", kernel, ", gamma =", gamma,
                                  ", degree =", degree, ", coef0 =", coef0,
                                  ", C =", C, ", Validation Accuracy = ",
                                  accuracy_mean, ", Validation log_loss = ",
                                  log_loss_mean)
                            if accuracy_mean > best_accuracy:
                                best_accuracy = accuracy_mean
                                best_C = C
                                best_kernel = kernel
                                best_gamma = gamma
                                best_degree = degree
                                best_coef0 = coef0
        # Select the best value found
        self.C = best_C
        self.kernel = best_kernel
        self.gamma = best_gamma
        self.degree = best_degree
        self.coef0 = best_coef0
        print("Best values : kernel =", self.kernel, ", gamma =",
              self.gamma, ", degree =", self.degree, ", coef0 =",
              self.coef0, ", C =", self.C)
        self.train(x_train, y_train)
        
if __name__ == '__main__':
    sm = SVM()
    dm = DataManager()
    dm.load_CSV("leaf-classification/train.csv","leaf-classification/test.csv")	
    x = dm.x_train
    y = dm.y_train
    sm.cross_validation(x, y)
    

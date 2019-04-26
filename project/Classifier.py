import numpy as np
import sklearn.metrics as skmetrics


# Super class for each classifier method
class Classifier:

    # Initialisation of the required parameters
    def __init__(self):
        pass

    def train(self, x_train, y_train):
        """
        Train the model with the x_train features array
        and the y_train one-hot labels
        Return : nothing
        """
        pass

    def test(self, x_test):
        """
        Test the model with the x_test features array
        Return : an array of predicted labels, as probabilities for each class
        """
        pass

    def cross_validation(self, x_train, y_train):
        """
        Train/test the model with different hyper parameters values,
        and select the best ones
        Return : nothing
        """
        pass

    def accuracy(self, predicted_labels, true_labels):
        """
        Compute the percentage of correct predictions over
        the total number of predictions (100*correct/total).
        The two inputs must be arrays of probabilities
        Return : the percentage of correct predictions, a float from 0 to 100
        """
        accuracy = 0.0
        # First version is for string or int labels
        if len(true_labels.shape) == 1:
            for predicted, true in zip(predicted_labels, true_labels):
                if predicted == true:
                    accuracy += 1

        # Second version if for one-hot labels
        else:
            for predicted, true in zip(predicted_labels, true_labels):
                if np.argmax(predicted) == np.argmax(true):
                    accuracy += 1

        return 100 * accuracy / predicted_labels.shape[0]

    def log_loss(self, predicted_labels, true_labels):
        """
        Compute the cross-entropy loss function of the prediction
        """
        log_loss = 0.0

        # First version is for string or int labels
        if len(true_labels.shape) == 1:
            for predicted, true in zip(predicted_labels, true_labels):
                if predicted != true:
                    # To follow the definition of Kaggle logloss
                    log_loss -= np.log(10e-15)
            return log_loss / predicted_labels.shape[0]
        # Second version if for one-hot labels
        else:
            return skmetrics.log_loss(true_labels, predicted_labels)

import csv
import numpy as np
import cv2
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import pdb

class MyDataManager():

    def __init__(self):
        # List of leaf features examples
        self.x_train = np.array([])
        # List of one-hot labels
        self.y_train = np.array([])
        # List of string labels, for every examples,
        # in the same order as y_train
        self.y_train_strings = np.array([])
        # List of integer labels, for every examples,
        # in the same order as y_train
        self.y_train_integer = np.array([])
        # Lists of leaf features examples without labels for testing
        self.x_test = np.array([])
        # Ids of all train examples, in the same order as x_train
        self.ids_test = np.array([])
        # Ids of all test examples, in the same order as x_test,
        # used to test on Kaggle
        self.ids_test = np.array([])
        # List of unique string labels as given by the CSV file,
        # used to test on Kaggle
        self.string_labels = np.array([])

    def string_to_onehot(self, s):
        """
        Convert a string label to a one hot vector
        """
        if isinstance(s, str):
            result = np.zeros((self.string_labels.shape[0]))
            if s in self.string_labels:
                result[np.where(self.string_labels == s)] = 1
            return result
        else:
            results = []
            for i in range(s.shape[0]):
                result = np.zeros((self.string_labels.shape[0]))
                if s[i] in self.string_labels:
                    result[np.where(self.string_labels == s[i])] = 1
                results.append(result)
            return np.array(results)

    def load_CSV(self, path_train, path_test):
        """
        Reads the csv files located at path_train and path_test,
        and stores them in the class variables defined above
        """
        print("")
        print("Loading training data...")
        # First part : training data
        with open(path_train) as csv_file_train:
            csv_file_train_reader = csv.reader(csv_file_train)
            # Skip header
            next(csv_file_train_reader)

            # List of the 192 features
            features = []
            # list of the leaf labels (string labels)
            # corresponding to the features
            y1 = []
            y2 = []
            y3 = []
            K = 50
            for row in csv_file_train_reader:
                y1.append(float(row[0]))
                y2.append(float(row[1]))
                y3.append(int(row[2]))
                features.append([float(feature) for feature in row[3:]])
            self.x_train = np.array(features)
            y3_train = np.array(y3)
            self.y_train_strings = np.array(y3_train)

            #select K features
            model = ExtraTreesClassifier()
            model.fit(self.x_train, self.y_train_strings)
            print("Important Features: ", model.feature_importances_) 
            feat_importances = pd.Series(model.feature_importances_)
            feat_importances.nlargest(K).plot(kind='barh')
            newIndex = feat_importances.nlargest(K).index
            self.x_train = self.x_train[:, newIndex]
            #plt.show()
            #pdb.set_trace()
            unique_labels = np.unique(y3_train)
            self.y_train = np.zeros((len(features), unique_labels.shape[0]))
            for l in range(len(y3)):
                one_hot_index = np.where(unique_labels == y3[l])[0]
                self.y_train[l][one_hot_index] = 1

            y_value = list(range(0, len(unique_labels)))
            y_dict = dict(zip(unique_labels, y_value))
            temp = pd.Series(y3)
            self.y_train_integer = temp.map(y_dict)


            #pdb.set_trace()
            '''
            self.y_train_strings = np.array(labels_string)
            # Establish a link between leaf names and unique assigned ids
            unique_labels = np.unique(labels_string)
            self.string_labels = unique_labels
            # Leaf labels (but this time converted to one-hot vectors)
            # corresponding to the features
            self.y_train = np.zeros((len(features), unique_labels.shape[0]))
            for l in range(len(labels_string)):
                one_hot_index = np.where(unique_labels == labels_string[l])[0]
                self.y_train[l][one_hot_index] = 1   
            y_value = list(range(0, len(unique_labels)))
            y_dict = dict(zip(unique_labels, y_value))
            temp = pd.Series(labels_string)
            self.y_train_integer = temp.map(y_dict)
            '''

        print("-> " + str(self.x_train.shape[0]) + " training examples loaded")

        print("Loading testing data...")
        # Second part : testing data
        with open(path_test) as csv_file_test:
            csv_file_test_reader = csv.reader(csv_file_test)
            # Skip header
            next(csv_file_test_reader)

            # List of the 192 features
            features = []
            for row in csv_file_test_reader:
                features.append([float(feature) for feature in row[0:]])
            self.x_test = np.array(features)
        print("-> " + str(self.x_test.shape[0]) + " testing examples loaded")

    """
    Write test prediction in a CSV file
    """
    def write_CSV(self, path, pred_test):
        print("Writing CSV")
        with open(path, mode="w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            # Write header
            header = np.concatenate((np.array(["id"]), self.string_labels))
            writer.writerow(header)
            for i in range(self.x_test.shape[0]):
                # String labels
                if isinstance(pred_test[i], str):
                    strings_to_onehot = self.string_to_onehot(pred_test[i])
                    writer.writerow([self.ids_test[i]]+list(strings_to_onehot))
                # One-hot vectors
                elif len(pred_test.shape) != 1:
                    writer.writerow([self.ids_test[i]]+list(pred_test[i]))
                # Integer labels
                else:
                    one_hot = np.zeros(self.string_labels.shape[0])
                    one_hot[pred_test[i]] = 1
                    writer.writerow([self.ids_test[i]]+list(one_hot))

    """
    Center and normalize the columns of x_train and x_test
    """
    def center_normalize_data(self):
        # Train data
        mean = np.mean(self.x_train, axis=0)
        std = np.std(self.x_train, axis=0)
        self.x_train = (self.x_train - mean)/std
        # Test data
        mean = np.mean(self.x_test, axis=0)
        std = np.std(self.x_test, axis=0)
        self.x_test = (self.x_test - mean)/std
        print("Data is now centered and normalized")

    """
    Load leaf images from the path and extract
    features from them (width, length, ratio, ...).
    Those features are added in x_train and x_test.
    The folder must have every image. Raises an error if one image is missing.
    """
    def extract_features_images(self, path):
        # Training set
        widths = []
        heights = []
        ratios = []
        squares = []
        orientations = []
        for i in range(self.ids_train.shape[0]):
            img = cv2.imread(path+str(self.ids_train[i])+'.jpg', 0)
            height, width = img.shape[:2]
            widths.append(width)
            heights.append(height)
            ratios.append(width/height)
            squares.append(width*height)
            orientations.append(int(width > height))
        self.x_train = np.concatenate((self.x_train,
                                       np.array([widths]).T), axis=1)
        self.x_train = np.concatenate((self.x_train,
                                       np.array([heights]).T), axis=1)
        self.x_train = np.concatenate((self.x_train,
                                       np.array([ratios]).T), axis=1)
        self.x_train = np.concatenate((self.x_train,
                                       np.array([squares]).T), axis=1)
        self.x_train = np.concatenate((self.x_train,
                                       np.array([orientations]).T), axis=1)
        # Testing set
        widths = []
        heights = []
        ratios = []
        squares = []
        orientations = []
        for i in range(self.ids_test.shape[0]):
            img = cv2.imread(path+str(self.ids_test[i])+'.jpg', 0)
            height, width = img.shape[:2]
            widths.append(width)
            heights.append(height)
            ratios.append(width/height)
            squares.append(width*height)
            orientations.append(int(width > height))
        self.x_test = np.concatenate((self.x_test,
                                      np.array([widths]).T), axis=1)
        self.x_test = np.concatenate((self.x_test,
                                      np.array([heights]).T), axis=1)
        self.x_test = np.concatenate((self.x_test,
                                      np.array([ratios]).T), axis=1)
        self.x_test = np.concatenate((self.x_test,
                                      np.array([squares]).T), axis=1)
        self.x_test = np.concatenate((self.x_test,
                                      np.array([orientations]).T), axis=1)
        print("Images features have been added")

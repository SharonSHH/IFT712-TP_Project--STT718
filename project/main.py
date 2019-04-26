from Classifier import Classifier
from DataManager import DataManager
from MyDataManager import MyDataManager
from AdaBoost import AdaBoost
from RandomForest import RandomForest
from SVM import SVM
from kNN import kNN
from Perceptron import Perceptron
from Ridge import Ridge
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2
import sys
import warnings


warnings.filterwarnings('ignore')
"""
Usage : main.py [method] [cv] [cn] [f] [dp] [w]
method can be AdaBoost, RandomForest, SVM, kNN, Perceptron, Ridge
cv can be 1 for cross-validation or 0 otherwise
cn can be 1 to use centered/normalize data, or 0 otherwise
f can be 1 to use images features, or 0 otherwise
dp can be 1 to use data pruning (i.e. only train and test on a subset of useful features), or 0 otherwise
w can be 1 to write a submission file or 0 otherwise
"""

# Check number of parameters
if len(sys.argv) != 7:
    print("Wrong arguments")
    print("Usage : main.py [method] [cv] [cn] [f] [w]")
    print("method can be AdaBoost, RandomForest, SVM, kNN, Perceptron, Ridge")
    print("cv can be 1 for cross-validation or 0 otherwise")
    print("cn can be 1 to use centered/normalize data, or 0 otherwise")
    print("f can be 1 to use images features, or 0 otherwise")
    print("dp can be 1 to use data pruning (i.e. only train and test on a subset of useful features), or 0 otherwise")
    print("w can be 1 to write a submission file or 0 otherwise")
    exit()

method = sys.argv[1]
cv = sys.argv[2]
cn = sys.argv[3]
f = sys.argv[4]
dp = sys.argv[5]
w = sys.argv[6]

# If method is incorrect, exit
if method not in ["AdaBoost", "RandomForest", "SVM",
                  "kNN", "Perceptron", "Ridge"]:
    print("Wrong methods. Possible values:")
    print("AdaBoost, RandomForest, SVM, kNN, Perceptron, Ridge")
    exit()

# If cv is incorrect, exit
if cv not in ['0', '1']:
    print("Wrong cross-validation parameter. Possible values:")
    print("1 = cross-validation, 0 = no cross-validation")
    exit()

# If cn is incorrect, exit
if cn not in ['0', '1']:
    print("Wrong center/normalize parameter. Possible values:")
    print("1 = center/normalize data, 0 = raw data")
    exit()

# If f is incorrect, exit
if f not in ['0', '1']:
    print("Wrong images features parameter. Possible values:")
    print("1 = add image features, 0 = no image features")
    exit()

# If dp is incorrect, exit
if dp not in ['0', '1']:
    print("Wrong data pruning parameter. Possible values:")
    print("1 = add image features, 0 = no image features")
    exit()

# If w is incorrect, exit
if w not in ['0', '1']:
    print("Wrong write parameters. Possible values:")
    print("1 = write submission file, 0 = no submission file")
    exit()

# Load leaf data
dm = MyDataManager()
#dm = DataManager()
dm.load_CSV("leaf-classification/train.csv",
            "leaf-classification/test.csv")
# Extract image features if needed
if f == '1':
    dm.extract_features_images("leaf-classification/images/")
# Center/normalize if needed
if cn == '1':
    dm.center_normalize_data()

x_train = dm.x_train
x_test = dm.x_test
y_train = None
classifier = None
# Select classifier, and corresponding labels
if method == "AdaBoost":
    y_train = dm.y_train_integer
    classifier = AdaBoost()
elif method == "RandomForest":
    y_train = dm.y_train_integer
    SelectKBest(chi2, k=30).fit(x_train, y_train)
    classifier = RandomForest()
elif method == "SVM":
    y_train = dm.y_train_strings
    classifier = SVM()
elif method == "kNN":
    y_train = dm.y_train_strings
    classifier = kNN()
elif method == "Ridge":
    y_train = dm.y_train
    classifier = Ridge()
elif method == "Perceptron":
    y_train = dm.y_train
    classifier = Perceptron()

# If cv=1, cross-validate
# If cv=0, train with default parameters
if cv == '1':
    print("Cross-validating model " + method)
    classifier.cross_validation(x_train, y_train)
else:
    print("Training model " + method + " with default hyperparameters")
    classifier.train(x_train, y_train)

# If dp = 1, features pruning
if dp == '1':
    # There are 192 features, so we decide
    # to remove up to a 100 of them
    print("Performing feature pruning")
    rfe = RFE(classifier.model, 150)
    rfe.fit(x_train, y_train)
    support = rfe.get_support(indices=True)
    x_train = x_train[:,support]
    x_test = x_test[:,support]
    classifier.train(x_train, y_train)

# Finally, assessing train accuracy
print("Evaluating model")
train_predict = classifier.test(x_train)
train_accuracy = classifier.accuracy(train_predict, y_train)
train_logloss = classifier.log_loss(train_predict, y_train)
print("Training accuracy =", train_accuracy, "%")
print("Training logloss =", train_logloss)

# If needed, write submission file
if w == '1':
    dm.write_CSV("test_results.csv", classifier.test(x_test))

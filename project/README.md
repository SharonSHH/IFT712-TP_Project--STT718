# IFT712 Project
## Thibault Duhamel / Heng Shi

## Requirements
- python3.5
- sklearn
- numpy
- openCV (cv2)
- pandas
- csv

## Intructions
You may run the main.py file with the following command and arguments:
```bash
python3.5 main.py [method] [cv] [cn] [f] [dp] [w]
```
- method is the classifier to use. It can be AdaBoost, RandomForest, SVM, kNN, Perceptron, Ridge
- cv must be either 1 to use cross-validation or 0 otherwise
- cn must be either 1 to center/normalize data or 0 otherwise
- f must be either 1 to use image features or 0 otherwise
- dp must be either 1 to use data pruning (consider a subset of features), or 0 otherwise
- w must be either 1 to write a submission file for Kaggle or 0 otherwise

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import argparse

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from tabulate import tabulate

from models import logLoss, svm, decisionTree, randomForest, dense_network
from preprocessing import get_preprocessor

def log_results(y_test, **predictions):
  print('Logging Results for the test set')
  results = []

  for model, pred in predictions.items():
     results.append([model, recall_score(y_test, pred),
                     precision_score(y_test, pred), 
                     f1_score(y_test, pred),
                     roc_auc_score(y_test, pred)])
     headers=["Model", "Recall", "Precision", "F1", "ROC AUC"]
     print(tabulate(results, headers, tablefmt="grid"))


def parse_arguments(sys_argv): 
    parser = argparse.ArgumentParser() 
    print('Parsing arguments...')
    parser.add_argument('--model',
                        help='Select the model to use',
                        choices=['all', 'LogLoss', 'DecTree', 'SVM', 'RF', 'DNN'],
                        default='all',
                        type=str)
    parser.add_argument('--best_parameters',
                        help='Use best parameters generated during testing',
                        default=False,
                        type=bool)
    parser.add_argument('--save_model',
                        help='Save the parameters for the trained models',
                        default=False, 
                        type=bool)
    
    return parser.parse_args(sys_argv)

def main():
    print('in main')
    print(f'sklearn version:{sklearn.__version__}')
    args = parse_arguments(sys.argv[1:])

    print('Loading dataset...')
    dataset = pd.read_csv('Dataset/in-vehicle-coupon-recommendation.csv')
    training_dataset, labels = dataset.drop('Y', axis=1), dataset['Y']
    X_train, X_test, y_train, y_test = train_test_split(training_dataset, labels, test_size=0.2, random_state=42)

    preprocessor = get_preprocessor()
    predictions = dict()
    match args.model:
        case 'LogLoss':
          predictions['LogLoss'] = logLoss(X_train, y_train, X_test, preprocessor)
        case 'SVM': 
          predictions['SVM'] = svm(X_train, y_train, X_test, preprocessor)
        case 'RF':
          predictions['RF'] = randomForest(X_train, y_train, X_test, preprocessor)
        case 'DecTree':
          predictions['DecTree'] = decisionTree(X_train, y_train, X_test, preprocessor)
        case 'DNN':
          predictions['DNN'] = dense_network(X_train, y_train, X_test, preprocessor)
        case 'all':
            predictions['LogLoss'] = logLoss(X_train, y_train, X_test, preprocessor)
            predictions['SVM'] = svm(X_train, y_train, X_test, preprocessor)
            predictions['RF'] = randomForest(X_train, y_train, X_test, preprocessor)
            predictions['DecTree'] = decisionTree(X_train, y_train, X_test, preprocessor)
            predictions['DNN'] = dense_network(X_train, y_train, X_test, preprocessor)
        case _: 
          raise Exception(f'model:{args.model} not found')
    
    log_results(y_test, **predictions)


if __name__ == '__main__':
  main()
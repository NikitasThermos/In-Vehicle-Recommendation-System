import sys

import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

from models import logLoss, svm, decisionTree, randomForest, dense_network

def log_results(model, y_true, y_pred):
  results = []
  conf_matrix = confusion_matrix(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  roc_auc = roc_auc_score(y_true, y_pred)
  print(f'confusion matrix for {model}:\n {conf_matrix}')
  print(f'Precision Score: {precision:.2f}')
  print(f'Recall Score: {recall:.2f}')
  print(f'F1 Score: {f1:.2f}')
  print(f'ROC_AUC Score: {roc_auc:.2f}')
  results.append([model, precision, recall, f1, roc_auc])

def parse_arguments(sys_argv): 
    parser = argparse.ArgumentParser() 

    parser.add_argument('--model',
                        help='Select the model to use',
                        choices=['all', 'LogLoss', 'SVM', 'RF', 'DNN'],
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
    args = parse_arguments(sys.argv[1:])

    dataset = pd.read_csv('Dataset/in-vehicle-coupon-recommendation.csv')
    training_dataset, labels = dataset.drop('Y', axis=1), dataset['Y']
    X_train, X_test, y_train, y_test = train_test_split(training_dataset, labels, test_size=0.2, random_state=42)

    predictions = dict()
    match args.model:
        case 'LogLoss':
          predictions['LogLoss'] = logLoss(X_train, y_train, X_test)
        case 'SVM': 
          predictions['SVM'] = svm(X_train, y_train, X_test)
        case 'RF':
          predictions['RF'] = randomForest(X_train, y_train, X_test)
        case 'DecTree':
          predictions['DecTree'] = decisionTree(X_train, y_train, X_test)
        case 'DNN':
          predictions['DNN'] = dense_network(X_train, y_train, X_test)
        case 'all':
            predictions['LogLoss'] = logLoss(X_train, y_train, X_test)
            predictions['SVM'] = svm(X_train, y_train, X_test)
            predictions['RF'] = randomForest(X_train, y_train, X_test)
            predictions['DecTree'] = decisionTree(X_train, y_train, X_test)
            predictions['DNN'] = dense_network(X_train, y_train, X_test)
        case _: 
          raise Exception(f'model:{args.model} not found')
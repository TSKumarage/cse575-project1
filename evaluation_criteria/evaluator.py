"""
Authors : Kumarage Tharindu
Class : CSE 575
Organization : ASU CIDSE
Project : SML Project 1
Task : Define different evaluation metrics

"""

import numpy as np


def mean_absolute_error(predict, target):

    mae = (np.sum(np.absolute(predict, target)))/len(target)

    return mae


def mean_squared_error(predict, target):

    mse = (np.square(predict - target)).mean()

    return mse


def binary_classification_performance(predict, target):

    predict = predict.astype(int)
    target = target.astype(int)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(target)):
        if target[i] == predict[i]:
            if predict[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if predict[i] == 1:
                fp += 1
            else:
                fn += 1

    print("Accuracy of the classification: ", tp/len(target))
    print("Precision of the classification: ", tp/(tp+fp))
    print("Recall of the classification: ", tp / (tp + fn))


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

    accuracy = (tp+tn)/len(target)
    precision = tp/(tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision*recall)/(precision+recall)
    F1_2 = 2 * ((1-precision)*(1-recall))/((1-precision)+(1-recall))

    print("{:^50s}".format("-------------Binary Classification Report-------------"))
    print()
    metric_list = ["Accuracy", "Precision", "Recall", "F1"]
    digits_data = [["8", accuracy, precision, recall, F1],
                   ["7", 1-accuracy, 1 - precision, 1-recall, F1_2]]

    row_format = '{0:<10} {1:>10} {2:>10} {3:>10} {4:>10}'
    print(row_format.format("Digit", *metric_list))
    for row in digits_data:
        print('{0:<10} {1:>10.2f} {2:>10.2f} {3:>10.2f} {4:>10.2f}'.format(*row))

    print()
    print("{:^50}".format("-------------End of Report-------------"))


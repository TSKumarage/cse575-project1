"""
Authors : Kumarage Tharindu
Class : CSE 575
Organization : ASU CIDSE
Project : SML Project 1
Task : Extract Mean and Std features

"""


import os
import scipy.io
import numpy as np


def main():
    data_mat = open_data()

    train = data_mat['trX']
    train_lbl = data_mat['trY']

    test = data_mat['tsX']
    test_lbl = data_mat['tsY']

    train_x = feature_extract(train)
    test_x = feature_extract(test)

    print(train_x)
    print(test_x)


def open_data():

    code_dir = os.path.dirname(__file__)  # absolute dir

    rel_path = os.path.join(code_dir, "Data", "mnist_data.mat")

    data = scipy.io.loadmat(rel_path)

    return data


def feature_extract(data):

    preprocessed_data = list()

    for image in data:
        features = list()
        features.append(np.mean(image))
        features.append(np.std(image))
        preprocessed_data.append(features)

    preprocessed_data = np.array(preprocessed_data)

    return preprocessed_data


if __name__ == '__main__':
    main()

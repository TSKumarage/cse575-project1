"""
Authors : Kumarage Tharindu
Class : CSE 575
Organization : ASU CIDSE
Project : SML Project 1
Task : Logistic Regression Classifier Implementation

"""

import numpy as np
import preprocess.feature_extraction as extrct
import evaluation_criteria.evaluator as eval


def main():
    data_mat = extrct.open_data()

    train = data_mat['trX']
    train_lbl = data_mat['trY'].reshape(data_mat['trY'].shape[1])

    test = data_mat['tsX']
    test_lbl = data_mat['tsY'].reshape(data_mat['tsY'].shape[1])

    train_x = extrct.feature_extract(train)
    test_x = extrct.feature_extract(test)

    print(train_x)
    print(test_x)

    print(train_lbl.shape)

    model = LogisticRegression()

    model.train(x=train_x, y=train_lbl, epochs=100)

    predict = model.predict(x=test_x)

    print(predict)


class LogisticRegression:
    def __init__(self, weight_init='zero', classifier_threshold=0.5):
        self.weight_init = weight_init
        self.optimized_weights = None
        self.classifier_threshold = classifier_threshold

    def logit_sigmoid(self, x, w):

        regression_value = np.dot(x, w)

        sigmoid = 1.0 / (1 + np.exp(-regression_value))

        return sigmoid

    def log_likelihood(self, x, y, weights):

        regression_value = np.dot(x, weights)

        ll = np.sum(y * regression_value - np.log(1 + np.exp(regression_value)))

        return ll

    def weight_initialize(self, shape=1):

        if self.weight_init == 'random':
            return np.random.rand(shape)

        if self.weight_init == 'zero':
            return np.zeros(shape)

    def train(self, x, y, epochs, learning_rate=0.00005, display=True):

        weights = self.weight_initialize(x.shape[1])

        for iteration in range(epochs):

            logit_values = self.logit_sigmoid(x, weights)  # Prediction value using sigmoid function for each row in x

            error = y - logit_values          # Error between the prediction and the target label

            gradient = np.dot(x.T, error)    # Calculate gradient of the

            weights += learning_rate*gradient              # gradient ascent

            if display:  # Print the error in each iteration
                print("epochs: ", iteration, " -- Log likelihood: ", self.log_likelihood(x, y, weights))

        self.optimized_weights = weights

    def predict(self, x):

        logit_values = self.logit_sigmoid(x, self.optimized_weights)  # Prediction value using sigmoid function for each row in x

        prediction_list = []
        for prediction in logit_values:
            if prediction > self.classifier_threshold:
                prediction_list.append(1)
            else:
                prediction_list.append(0)

        return np.array(prediction_list)


if __name__ == '__main__':
    main()

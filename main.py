"""
Authors : Kumarage Tharindu
Class : CSE 575
Organization : ASU CIDSE
Project : SML Project 1
Task : Main script to run all the tasks of the project

"""

import preprocess.feature_extraction as extrct
import evaluation_criteria.evaluator as eval
import logistic_regression.logit_classifier as LR
import naive_bayes.naive_bayes_classifier as NB


def main():
    data_mat = extrct.open_data()  # Open the given files

    train = data_mat['trX']
    train_lbl = data_mat['trY'].reshape(data_mat['trY'].shape[1])

    test = data_mat['tsX']
    test_lbl = data_mat['tsY'].reshape(data_mat['tsY'].shape[1])

    train_x = extrct.feature_extract(train)  # Feature extraction on training data
    test_x = extrct.feature_extract(test)  # Feature extraction on test data

    model = NB.NaiveBayes()  # Create a Naive Bayes model object

    model.train(x=train_x, y=train_lbl)  # Estimates densities

    predict = model.predict(x=test_x)  # Predict class for test data

    eval.binary_classification_performance(predict=predict, target=test_lbl)  # Print accuracy values

    print()

    model = LR.LogisticRegression()  # Create a Logistic regression model object

    model.train(x=train_x, y=train_lbl, learning_rate=0.0001, epochs=1000, display=False)  # Find optimum values for weights

    predict = model.predict(x=test_x)  # Predict class for test data

    eval.binary_classification_performance(predict=predict, target=test_lbl)  # Print accuracy values

if __name__ == '__main__':
    main()

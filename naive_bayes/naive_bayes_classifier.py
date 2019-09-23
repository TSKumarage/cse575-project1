"""
Authors : Kumarage Tharindu
Class : CSE 575
Organization : ASU CIDSE
Project : SML Project 1
Task : Naive Bayes Classifier Implementation

"""

import numpy as np
import math
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

    model = NaiveBayes()

    model.train(x=train_x, y=train_lbl)

    predict = model.predict(x=test_x)

    eval.binary_classification_performance(predict=predict, target=test_lbl)


class NaiveBayes:
    def __init__(self):
        self.density_parameters = dict()

    def density_estimation(self, x, y):
        # Estimate the densities of the given data
        class_labels = np.unique(y)

        for class_val in class_labels:  # For each class estimate the density parameters

            partition = []
            for i in range(x.shape[0]): # Extract the training samples for a given class
                if y[i] == class_val:
                    partition.append(x[i])

            partition = np.array(partition)

            feature_density_params = {}

            for i in range(x.shape[1]):
                feature_name = "x" + str(i+1)
                density = dict()

                density['mean'] = np.mean(partition[:, i])
                density['std'] = np.std(partition[:, i])

                feature_density_params[feature_name] = density

            cov = self.covariance_matrix(partition)

            feature_density_params['covariance_matrix'] = cov

            #  prior probability estimation for the class - P(y=class)
            feature_density_params['class_prior_probability'] = len(partition)/len(x)

            self.density_parameters[class_val] = feature_density_params

    def covariance_matrix(self, data_mat, independent_vars=True):
        # Calculate the covariance matrix of the given features

        cov = np.zeros((data_mat.shape[1], data_mat.shape[1]))
        if independent_vars:  # Since the covariance matrix is diagonal due to Naive assumption

            for i in range(data_mat.shape[1]):
                cov[i][i] = np.std(data_mat[:, i])

        else:  # For the completeness of the program Cov estimation without Naive assumption
            cov = np.cov(data_mat)

        return cov

    def show_data_density(self):
        # Display the data density parameters
        for class_lbl in self.density_parameters:
            if class_lbl == 1:
                digit = 8
            else:
                digit = 7

            print("Class: ", class_lbl)

            print("---------- Density Parameters of the digit ", digit, "----------")
            print()

            feature_densities = self.density_parameters[class_lbl]
            print("Class prior probability - P(class={}):".format(digit),
                  "{:.2f}".format(feature_densities['class_prior_probability']))
            print()

            row_format = '{0:<10} {1:>10} {2:>10}'
            param_list = ["Mean", "Stdev"]
            print(row_format.format("Feature", *param_list))

            feature_list = []
            for feature in sorted(feature_densities):

                if feature not in ['covariance_matrix', 'class_prior_probability']:
                    feature_list.append(feature)
                    row_format = '{0:<10} {1:>10.3f} {2:>10.3f}'
                    print(row_format.format(feature, feature_densities[feature]['mean'],
                                            feature_densities[feature]['std']))

            print()
            print("Covariance Matrix..")

            row_format = "{:>15}" * (len(feature_list) + 1)
            print(row_format.format("", *feature_list))

            row_format = "{:>15}" + ("{:>15.3f}" * len(feature_list))
            for feature, row in zip(feature_list, feature_densities['covariance_matrix']):
                print(row_format.format(feature, *row))
            print()

    def train(self, x, y):
        # Model training

        print("Training Naive Bayes Classifier..")

        print("Estimating data densities")

        self.density_estimation(x=x, y=y)  # Populate relevant densities

        self.show_data_density()

    def predict(self, x):
        # Predict the class for a given sample(s)

        prediction_list = []

        for x_i in x:

            prediction = 0
            max_class_probability = 0

            for class_lbl in self.density_parameters:

                feature_densities = self.density_parameters[class_lbl]

                mu = list()

                for feature in feature_densities:
                    if feature not in ['covariance_matrix', 'class_prior_probability']:
                        mu.append(feature_densities[feature]['mean'])

                mu = np.array(mu)

                # Calculate joint conditional probability P(x1, x2|y=class)
                # Since the covariance mat is diagonal this equals to product of
                # features conditional probabilities given tha class
                conditional_probability = self.multivariate_pdf(x_i, mu, feature_densities['covariance_matrix'])

                # Estimate the posterior probability P(y=class|x_i)
                class_probability = conditional_probability*feature_densities['class_prior_probability']

                if class_probability > max_class_probability:
                    max_class_probability = class_probability
                    prediction = class_lbl

            prediction_list.append(prediction)

        return np.array(prediction_list)

    def multivariate_pdf(self, x, mu, cov, k=2):
        # Standard multivariate normal distribution PDF matrix form

        numerator = np.exp((np.dot(np.dot((x-mu), np.linalg.inv(cov)), (x-mu).T))/-2)

        denominator = math.sqrt(math.pow((2*math.pi), k)*np.linalg.det(cov))

        return numerator/denominator


if __name__ == '__main__':
    main()






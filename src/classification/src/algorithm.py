# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
from numpy import exp, ndarray, pi, sqrt
from sklearn import naive_bayes, svm, tree


#############
# Interface #
#############

def create_naive_bayes(data, target):
    """
    :param data {DataFrame} data set
    :param target {Series} target set
    :return {GaussianNB} trained model
    """
    return naive_bayes.GaussianNB().fit(data, target)


def predict_naive_bayes(clf, data):
    """
    :param clf {GaussianNB} trained model
    :param data {DataFrame} data set
    :return {ndarray} prediction set
    """
    return clf.predict(data)


def create_decision_tree(data, target):
    """
    :param data {DataFrame} data set
    :param target {Series} target set
    :return {DecisionTreeClassifier} trained model
    """
    return tree.DecisionTreeClassifier().fit(data, target)


def predict_decision_tree(clf, data):
    """
    :param clf {DecisionTreeClassifier} trained model
    :param data {DataFrame} data set
    :return {ndarray} prediction set
    """
    return clf.predict(data)


def create_svm(data, target):
    """
    :param data {DataFrame} data set
    :param target {Series} target set
    :return {SVC} trained model
    """
    return svm.SVC(gamma='scale').fit(data, target)


def predict_svm(clf, data):
    """
    :param clf {SVC} trained model
    :param data {DataFrame} data set
    :return {ndarray} prediction set
    """
    return clf.predict(data)


def create_naive_bayes_diy(data, target):
    """
    :param data {DataFrame} data set
    :param target {Series} target set
    :return {GaussianNB} trained model
    """
    return GaussianNB().fit(data, target)


def predict_naive_bayes_diy(clf, data):
    """
    :param clf {GaussianNB} trained model
    :param data {DataFrame} data set
    :return {ndarray} prediction set
    """
    return clf.predict(data)


##############
# GaussianNB #
##############

class GaussianNB:
    def __init__(self):
        self.class_number = None  # Number of classes
        self.prior = None         # Prior probability
        self.avgs = None          # Averages of training set
        self.vars = None          # Variances of training set

    @staticmethod
    def __get_prior(target: ndarray) -> ndarray:
        """ Calculate prior probability of different classes
        :param target {ndarray} target values
        :return {ndarray}
        """
        count = Counter(target)
        prior = np.array([count[i] / len(target) for i in range(len(count))])
        return prior

    def __get_avgs(self, data: ndarray, target: ndarray) -> ndarray:
        """ Calculate averages of training set
        :param data {ndarray} data set
        :param target {ndarray} target set
        :return {ndarray}
        """
        return np.array([data[target == i].mean(axis=0) for i in range(self.class_number)])

    def __get_vars(self, data: ndarray, target: ndarray) -> ndarray:
        """ Calculate variances of training set
        :param data {ndarray} data set
        :param target {ndarray} target set
        :return {ndarray}
        """
        return np.array([data[target == i].var(axis=0) for i in range(self.class_number)])

    def __get_likelihood(self, sample: ndarray) -> ndarray:
        """ Calculate likelihood
        :param sample {ndarray} sample of data set
        :return {ndarray}
        """
        return (1 / sqrt(2 * pi * self.vars) * exp(-(sample - self.avgs) ** 2 / (2 * self.vars))).prod(axis=1)

    def __get_posterior(self, data: ndarray) -> ndarray:
        """ Calculate posterior probability of data set
        :param data {ndarray} data set
        :return {ndarray} probabilities of different classes
        """
        likelihood = np.apply_along_axis(self.__get_likelihood, axis=1, arr=data)
        posterior = self.prior * likelihood
        posterior_sum = posterior.sum(axis=1)
        return posterior / posterior_sum[:, None]

    def fit(self, data: ndarray, target: ndarray):
        """ Create a Gauss naive bayes classifier
        :param data {ndarray} data set
        :param target {ndarray} target set
        :return {GaussianNB}
        """
        # Calculate prior probability
        self.prior = self.__get_prior(target)
        self.class_number = len(self.prior)
        # Calculate the average
        self.avgs = self.__get_avgs(data, target)
        # Calculate the variance
        self.vars = self.__get_vars(data, target)
        return self

    def predict(self, data: ndarray) -> ndarray:
        """ Get the prediction of data set
        :param data {ndarray} data set
        :return {ndarray} prediction set
        """
        # Choose the class which has the maximum probability
        return self.__get_posterior(data).argmax(axis=1)

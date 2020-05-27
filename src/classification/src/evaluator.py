# -*- coding: utf-8 -*-
from algorithm import *
from sklearn.model_selection import KFold

DEFAULT_SPLIT_NUMBER = 10
kf = KFold(n_splits=DEFAULT_SPLIT_NUMBER, shuffle=True)


#############
# Interface #
#############
def evaluate_naive_bayes(data, target):
    """
    :param data {DataFrame} data set
    :param target {Series} target set
    :return {dictionary} evaluate result
    """
    print('[Evaluate] Run evaluate_naive_bayes()...')
    result = {'accuracy': 0, 'recall': 0, 'precision': 0, 'f_measure': 0}
    for train, test in kf.split(data):
        data_train, data_test, target_train, target_test = data[train], data[test], target[train], target[test]
        clf = create_naive_bayes(data_train, target_train)
        target_predicted = predict_naive_bayes(clf, data_test)
        result['accuracy'] += get_accuracy(target_test, target_predicted)
        result['recall'] += get_recall(target_test, target_predicted)
        result['precision'] += get_precision(target_test, target_predicted)
        result['f_measure'] += get_f_measure(target_test, target_predicted)
    result['accuracy'] /= DEFAULT_SPLIT_NUMBER
    result['recall'] /= DEFAULT_SPLIT_NUMBER
    result['precision'] /= DEFAULT_SPLIT_NUMBER
    result['f_measure'] /= DEFAULT_SPLIT_NUMBER
    return result


def evaluate_decision_tree(data, target):
    """
    :param data {DataFrame} data set
    :param target {Series} target set
    :return {dictionary} evaluate result
    """
    print('[Evaluate] Run evaluate_decision_tree()...')
    result = {'accuracy': 0, 'recall': 0, 'precision': 0, 'f_measure': 0}
    for train, test in kf.split(data):
        data_train, data_test, target_train, target_test = data[train], data[test], target[train], target[test]
        clf = create_decision_tree(data_train, target_train)
        target_predicted = predict_decision_tree(clf, data_test)
        result['accuracy'] += get_accuracy(target_test, target_predicted)
        result['recall'] += get_recall(target_test, target_predicted)
        result['precision'] += get_precision(target_test, target_predicted)
        result['f_measure'] += get_f_measure(target_test, target_predicted)
    result['accuracy'] /= DEFAULT_SPLIT_NUMBER
    result['recall'] /= DEFAULT_SPLIT_NUMBER
    result['precision'] /= DEFAULT_SPLIT_NUMBER
    result['f_measure'] /= DEFAULT_SPLIT_NUMBER
    return result


def evaluate_svm(data, target):
    """
    :param data {DataFrame} data set
    :param target {Series} target set
    :return {dictionary} evaluate result
    """
    print('[Evaluate] Run evaluate_svm()...')
    result = {'accuracy': 0, 'recall': 0, 'precision': 0, 'f_measure': 0}
    for train, test in kf.split(data):
        data_train, data_test, target_train, target_test = data[train], data[test], target[train], target[test]
        clf = create_svm(data_train, target_train)
        target_predicted = predict_svm(clf, data_test)
        result['accuracy'] += get_accuracy(target_test, target_predicted)
        result['recall'] += get_recall(target_test, target_predicted)
        result['precision'] += get_precision(target_test, target_predicted)
        result['f_measure'] += get_f_measure(target_test, target_predicted)
    result['accuracy'] /= DEFAULT_SPLIT_NUMBER
    result['recall'] /= DEFAULT_SPLIT_NUMBER
    result['precision'] /= DEFAULT_SPLIT_NUMBER
    result['f_measure'] /= DEFAULT_SPLIT_NUMBER
    return result


def evaluate_naive_bayes_diy(data, target):
    """
    :param data {DataFrame} data set
    :param target {Series} target set
    :return {dictionary} evaluate result
    """
    print('[Evaluate] Run evaluate_naive_bayes_diy()...')
    result = {'accuracy': 0, 'recall': 0, 'precision': 0, 'f_measure': 0}
    for train, test in kf.split(data):
        data_train, data_test, target_train, target_test = data[train], data[test], target[train], target[test]
        clf = create_naive_bayes_diy(data_train, target_train)
        target_predicted = predict_naive_bayes_diy(clf, data_test)
        result['accuracy'] += get_accuracy(target_test, target_predicted)
        result['recall'] += get_recall(target_test, target_predicted)
        result['precision'] += get_precision(target_test, target_predicted)
        result['f_measure'] += get_f_measure(target_test, target_predicted)
    result['accuracy'] /= DEFAULT_SPLIT_NUMBER
    result['recall'] /= DEFAULT_SPLIT_NUMBER
    result['precision'] /= DEFAULT_SPLIT_NUMBER
    result['f_measure'] /= DEFAULT_SPLIT_NUMBER
    return result


###############
# Calculation #
###############
def get_accuracy(y_actual, y_predicted):
    """
    :param y_actual {list} actual y
    :param y_predicted {list} predicted y
    :return {float} accuracy
    """
    return sum(a == b for a, b in zip(y_actual, y_predicted)) / len(y_actual)


def get_recall(y_actual, y_predicted):
    """
    :param y_actual {list} actual y
    :param y_predicted {list} predicted y
    :return {float} recall
    """
    true_positive = sum(a and b for a, b in zip(y_actual, y_predicted))
    actual_positive = sum(y_actual)
    if actual_positive == 0:
        return 1
    return true_positive / actual_positive


def get_precision(y_actual, y_predicted):
    """
    :param y_actual {list} actual y
    :param y_predicted {list} predicted y
    :return {float} precision
    """
    true_positive = sum(a and b for a, b in zip(y_actual, y_predicted))
    predicted_positive = sum(y_predicted)
    if predicted_positive == 0:
        return 1
    return true_positive / predicted_positive


def get_f_measure(y_actual, y_predicted):
    """
    :param y_actual {list} actual y
    :param y_predicted {list} predicted y
    :return {float} F-measure
    """
    r = get_recall(y_actual, y_predicted)
    p = get_precision(y_actual, y_predicted)
    return (2 * r * p) / (r + p)

# -*- coding: utf-8 -*-
from algorithm import *
from collections import Counter
from timer import *
from math import log2

# 4 / 8 / 10
DEFAULT_CLUSTER_NUMBER = 4
# Family / Genus / Species
DEFAULT_TARGET_FEATURE = 'Family'


#############
# Interface #
#############
def evaluate_k_means(data, target):
    """
    :param data {DataFrame} data set
    :param target {DataFrame} target set
    :return {dictionary} evaluate result
    """
    print('[Evaluate] Run evaluate_k_means()...')
    # Get
    cluster_class_actual = target[DEFAULT_TARGET_FEATURE].values
    cluster_class_predicted = perform_k_means(data, DEFAULT_CLUSTER_NUMBER)
    cluster_total_number = len(cluster_class_predicted)
    # Count
    result = {'entropy': 0, 'purity': 0}
    for cluster_index in range(DEFAULT_CLUSTER_NUMBER):
        # Statistics
        cluster_point_array = [cluster_class_actual[i] for i, c in enumerate(cluster_class_predicted) if c == cluster_index]
        cluster_point_number = len(cluster_point_array)
        cluster_counter = Counter(cluster_point_array)
        # Calculation
        entropy = 0
        purity = 0
        for c in cluster_counter:
            pr = cluster_counter.get(c) / cluster_point_number
            entropy -= pr * log2(pr)
            purity = max(purity, pr)
        result['entropy'] += cluster_point_number / cluster_total_number * entropy
        result['purity'] += cluster_point_number / cluster_total_number * purity
    return result


def evaluate_spectral_clustering(data, target):
    """
    :param data {DataFrame} data set
    :param target {DataFrame} target set
    :return {dictionary} evaluate result
    """
    print('[Evaluate] Run evaluate_spectral_clustering()...')
    # Get
    cluster_class_actual = target[DEFAULT_TARGET_FEATURE].values
    cluster_class_predicted = perform_spectral_clustering(data, DEFAULT_CLUSTER_NUMBER)
    cluster_total_number = len(cluster_class_predicted)
    # Count
    result = {'entropy': 0, 'purity': 0}
    for cluster_index in range(DEFAULT_CLUSTER_NUMBER):
        # Statistics
        cluster_point_array = [cluster_class_actual[i] for i, c in enumerate(cluster_class_predicted) if c == cluster_index]
        cluster_point_number = len(cluster_point_array)
        cluster_counter = Counter(cluster_point_array)
        # Calculation
        entropy = 0
        purity = 0
        for c in cluster_counter:
            pr = cluster_counter.get(c) / cluster_point_number
            entropy -= pr * log2(pr)
            purity = max(purity, pr)
        result['entropy'] += cluster_point_number / cluster_total_number * entropy
        result['purity'] += cluster_point_number / cluster_total_number * purity
    return result


def evaluate_k_means_diy(data, target):
    """
    :param data {DataFrame} data set
    :param target {DataFrame} target set
    :return {dictionary} evaluate result
    """
    print('[Evaluate] Run evaluate_k_means_diy()...')
    # Get
    cluster_class_actual = target[DEFAULT_TARGET_FEATURE].values
    cluster_class_predicted = perform_k_means_diy(data.values, DEFAULT_CLUSTER_NUMBER)
    cluster_total_number = len(cluster_class_predicted)
    # Count
    result = {'entropy': 0, 'purity': 0}
    for cluster_index in range(DEFAULT_CLUSTER_NUMBER):
        # Statistics
        cluster_point_array = [cluster_class_actual[i] for i, c in enumerate(cluster_class_predicted) if c == cluster_index]
        cluster_point_number = len(cluster_point_array)
        cluster_counter = Counter(cluster_point_array)
        # Calculation
        entropy = 0
        purity = 0
        for c in cluster_counter:
            pr = cluster_counter.get(c) / cluster_point_number
            entropy -= pr * log2(pr)
            purity = max(purity, pr)
        result['entropy'] += cluster_point_number / cluster_total_number * entropy
        result['purity'] += cluster_point_number / cluster_total_number * purity
    return result

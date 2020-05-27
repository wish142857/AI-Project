# -*- coding: utf-8 -*-
import pandas as pd
from algorithm import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from timer import *


# 4 / 8 / 10
DEFAULT_CLUSTER_NUMBER = 4


#############
# Interface #
#############
def paint_k_means(data, path):
    """
    :param data {DataFrame} data set
    :param path {String} save file path
    """
    print('[Painter] Run paint_k_means()...')
    # Call
    cluster_class_predicted = perform_k_means(data, DEFAULT_CLUSTER_NUMBER)
    # Paint
    try:
        paint_plt(data, cluster_class_predicted, path)
    except FileNotFoundError:
        return '[Failure] File path not found: %s.' % path
    return '[Success] Picture saved in %s.' % path


def paint_spectral_clustering(data, path):
    """
    :param data {DataFrame} data set
    :param path {String} save file path
    """
    print('[Painter] Run paint_spectral_clustering()...')
    # Call
    cluster_class_predicted = perform_spectral_clustering(data, DEFAULT_CLUSTER_NUMBER)
    try:
        paint_plt(data, cluster_class_predicted, path)
    except FileNotFoundError:
        return '[Failure] File path not found: %s.' % path
    return '[Success] Picture saved in %s.' % path


def paint_k_means_diy(data, path):
    """
    :param data {DataFrame} data set
    :param path {String} save file path
    """
    print('[Painter] Run paint_k_means_diy()...')
    # Call
    cluster_class_predicted = perform_k_means_diy(data.values, DEFAULT_CLUSTER_NUMBER)
    try:
        paint_plt(data, cluster_class_predicted, path)
    except FileNotFoundError:
        return '[Failure] File path not found: %s.' % path
    return '[Success] Picture saved in %s.' % path


def paint_plt(data, target, path):
    """
    :param data {DataFrame} data set
    :param target {DataFrame} target set
    :param path {String} save file path
    """
    # Dimensionality reduction
    tsne = TSNE()
    tsne.fit_transform(data)
    data_2d = pd.DataFrame(tsne.embedding_, index=data.index)
    # Classification
    cluster_point_array_0 = [data_2d.values[i] for i, c in enumerate(target) if c == 0]
    cluster_point_array_1 = [data_2d.values[i] for i, c in enumerate(target) if c == 1]
    cluster_point_array_2 = [data_2d.values[i] for i, c in enumerate(target) if c == 2]
    cluster_point_array_3 = [data_2d.values[i] for i, c in enumerate(target) if c == 3]
    plt.clf()
    plt.plot(cluster_point_array_0, 'rv')
    plt.plot(cluster_point_array_1, 'go')
    plt.plot(cluster_point_array_2, 'b*')
    plt.plot(cluster_point_array_3, 'y+')
    plt.savefig(path)
    plt.show()
    return

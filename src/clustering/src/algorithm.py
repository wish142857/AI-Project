# -*- coding: utf-8 -*-
import random
from collections import Counter
from copy import deepcopy
from sklearn import cluster


#############
# Interface #
#############
def create_k_means(data, cluster_number):
    """
    :param data {DataFrame} data set
    :param cluster_number {int} the number of clusters
    :return {KMeans} trained model
    """
    return cluster.KMeans(n_clusters=cluster_number).fit(data)


def predict_k_means(clt, data):
    """
    :param clt {KMeans} trained model
    :param data {DataFrame} data set
    :return {ndarray} prediction set
    """
    return clt.predict(data)


def perform_k_means(data, cluster_number):
    """
    :param data {DataFrame} data set
    :param cluster_number {int} the number of clusters
    :return {ndarray} prediction set
    """
    return cluster.KMeans(n_clusters=cluster_number).fit_predict(data)


def create_spectral_clustering(data, cluster_number):
    """
    :param data {DataFrame} data set
    :param cluster_number {int} the number of clusters
    :return {SpectralClustering} trained model
    """
    return cluster.SpectralClustering(n_clusters=cluster_number).fit(data)


def perform_spectral_clustering(data, cluster_number):
    """
    :param data {DataFrame} data set
    :param cluster_number {int} the number of clusters
    :return {ndarray} prediction set
    """
    return cluster.SpectralClustering(n_clusters=cluster_number).fit_predict(data)


def create_k_means_diy(data, cluster_number):
    """
    :param data {DataFrame} data set
    :param cluster_number {int} the number of clusters
    :return {KMeans} trained model
    """
    return KMeans().fit(data, cluster_number)


def predict_k_means_diy(clt, data):
    """
    :param clt {KMeans} trained model
    :param data {DataFrame} data set
    :return {ndarray} prediction set
    """
    return clt.predict(data)


def perform_k_means_diy(data, cluster_number):
    """
    :param data {DataFrame} data set
    :param cluster_number {int} the number of clusters
    :return {ndarray} prediction set
    """
    return KMeans().fit_predict(data, cluster_number)


##########
# KMeans #
##########

def check_equality(X, Y, tol=1e-8) -> bool:
    """" Check whether two vectors are the same under tolerance
    :param X {list} vector-1
    :param Y {list} vector-2
    :param tol {float} tolerance
    :return {bool}
    """
    return (len(X) == len(Y)) and (all(abs(x - y) < tol for x, y in zip(X, Y)))


def check_convergence(M, N) -> bool:
    """ Check whether two center point vectors list are the same under tolerance
    :param M {list} list-1 of center point vectors
    :param N {list} list-2 of center point vectors
    :return {bool}
    """
    return all(check_equality(X, Y) for X, Y in zip(M, N))


def get_euclidean_distance(X, Y) -> float:
    """" Calculate the euclidean distance of two vectors
    :param X {list} vector-1
    :param Y {list} vector-2
    :return {float} euclidean distance
    """
    return ((X - Y) ** 2).sum() ** 0.5


def get_cosine_similarity(X, Y) -> float:
    """ Calculate the cosine similarity of two vectors
    :param X {list} vector-1
    :param Y {list} vector-2
    :return {float} cosine similarity
    """
    return (sum(x * y for x, y in zip(X, Y))) / ((sum(x ** 2 for x in X) * sum(y ** 2 for y in Y)) ** 0.5)


def binary_search(nums, target):
    """ Binary search target from array nums
    :param nums {list}
    :param target {float}
    :return {int} minimum index of the element in nums >= target
    """
    low = 0
    high = len(nums) - 1
    assert nums[low] <= target < nums[high], "Error in binary search!"
    while 1:
        mid = (low + high) // 2
        if target >= nums[mid] or mid == 0:
            low = mid + 1
        elif target < nums[mid - 1]:
            high = mid - 1
        else:
            break
    return mid


class KMeans(object):
    """KMeans class.
    Attributes:
        k {int} -- Number of cluster centers.
        n_features {int} -- Number of features.
        cluster_centers {list} -- 2d list with int or float.
        distance_fn {function} -- The function to measure the distance.
        cluster_samples_cnt {Counter} --  Count of samples in each cluster.
    """

    def __init__(self):
        self.k = None  # number of cluster centers
        self.feature_number = None  # number of features
        self.distance_function = None  # distance function
        self.cluster_centers = None  # cluster center points

    def __get_nearest_center(self, X, centers):
        """ Find the nearest center point of X (1d)
        :param X {list} vector
        :param centers {list} list of center point vectors
        :return {int} index of nearest cluster center point
        """
        return min(((i, self.distance_function(X, center)) for i, center in enumerate(centers)), key=lambda x: x[1])[0]

    def __get_nearest_centers(self, M, centers):
        """ Find the nearest center point of M (2d)
        :param M {list} list of vectors
        :param centers {list} list of center point vectors
        :return {list} indexes of nearest cluster center points
        """
        return [self.__get_nearest_center(X, centers) for X in M]

    def __init_cluster_centers(self, X, k):
        """ Generate initial cluster centers with K-means++
        :param X {list} data set
        :param k {int} number of cluster centers
        :return {list} list of initial cluster center point vectors
        """
        # Generate a single random center
        centers = [random.choice(X)]
        # Generate other random centers
        dist = [0 for _ in range(len(X))]
        for _ in range(1, k):
            total = 0.0
            for i, p in enumerate(X):
                dist[i] = self.distance_function(p, centers[self.__get_nearest_center(p, centers)])
                total += dist[i]
            total *= random.random()
            # Use Round Robin
            for i, d in enumerate(dist):
                total -= d
                if total <= 0:
                    centers.append(X[i])
                    break
        return centers

    def __update_cluster_centers(self, X, y, cluster_samples_cnt):
        """ Update cluster centers by the average of each cluster's samples
        :param X {list} data set
        :param y {list} index list of nearest centers
        :param cluster_samples_cnt {Counter} count of samples in each cluster
        :return {list} list of new cluster center point vectors
        """
        centers_new = [[0 for _ in range(self.feature_number)] for _ in range(self.k)]
        for p, nearest_center in zip(X, y):
            for i in range(self.feature_number):
                centers_new[nearest_center][i] += p[i] / cluster_samples_cnt[nearest_center]
        return centers_new

    def fit(self, X, k, fn=get_euclidean_distance, n_iter=100):
        """ Create K-Means model
        :param X {list} data set
        :param k {int} number of cluster centers
        :param fn {function} distance function
        :param n_iter {int} number of iterations
        :return {KMeans}
        """
        # Set number of cluster centers
        self.k = k
        # Set number of features
        self.feature_number = len(X[0])
        # Set distance function
        self.distance_function = fn
        # Initialization
        centers = self.__init_cluster_centers(X, self.k)
        # Cyclic iteration
        for i in range(n_iter):
            # Search the nearest cluster centers of X
            y = self.__get_nearest_centers(X, centers)
            # Count of samples in each cluster.
            cluster_samples_cnt = Counter(y)
            # Update cluster centers
            centers_new = self.__update_cluster_centers(X, y, cluster_samples_cnt)
            # Check convergence
            if check_convergence(centers, centers_new):
                break
            centers = deepcopy(centers_new)
        # Save cluster center points
        self.cluster_centers = centers
        return self

    def predict_single(self, Xi):
        """ Get the cluster center of Xi
        :param Xi {list} point vector
        :return {int} cluster center index
        """
        return self.__get_nearest_center(Xi, self.cluster_centers)

    def predict(self, X):
        """ Get the cluster center of X
        :param X {list} list of point vectors
        :return {list} list of cluster center indexes
        """
        return [self.predict_single(Xi) for Xi in X]

    def fit_predict(self, X, k):
        """ Create K-Means model and get the cluster center of X
        :param X {list} list of point vectors
        :param k {int} number of cluster centers
        :return {list} list of cluster center indexes
        """
        return self.fit(X, k).predict(X)

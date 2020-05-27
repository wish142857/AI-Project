# -*- coding: utf-8 -*-
import pandas as pd


def load_dataset(file_path):
    """
    :param file_path {String} input file's path
    :return dataset
    """
    # Define dataset & Load raw dataset from txt
    dataset = {'raw_data': pd.read_csv(file_path), 'data': None, 'target': None}
    # Extract target from raw dataset
    dataset['target'] = dataset['raw_data']['y']
    print('[Loader] Dataset read successful.')
    return dataset


def select_features_1(dataset):
    """ Select feature data from raw dataset (Combination-1)
    data -> dataset['data']
    """
    # Select feature data and get dummies
    dataset['data'] = pd.get_dummies(dataset['raw_data'][['age', 'job', 'education', 'default', 'balance', 'housing', 'loan']])
    print('[Loader] Select features combination-1.')
    return


def select_features_2(dataset):
    """ Select feature data from raw dataset (Combination-2)
    data -> dataset['data']
    """
    # Select feature data and get dummies
    dataset['data'] = pd.get_dummies(dataset['raw_data'][['job', 'contact', 'duration', 'pdays', 'previous', 'poutcome']])
    print('[Loader] Select features combination-2.')
    return

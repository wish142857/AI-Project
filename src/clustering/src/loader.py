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
    dataset['target'] = dataset['raw_data'][['Family', 'Genus', 'Species']]
    print('[Loader] Dataset read successful.')
    return dataset


def select_features_1(dataset):
    """ Select feature data from raw dataset (Combination-1)
    data -> dataset['data']
    """
    # Select feature data
    dataset['data'] = dataset['raw_data'][['MFCCs_17', 'MFCCs_18', 'MFCCs_19', 'MFCCs_20', 'MFCCs_21', 'MFCCs_22']]
    print('[Loader] Select features combination-1.')
    return


def select_features_2(dataset):
    """ Select feature data from raw dataset (Combination-2)
    data -> dataset['data']
    """
    # Select feature data
    dataset['data'] = dataset['raw_data'][['MFCCs_11', 'MFCCs_12', 'MFCCs_13', 'MFCCs_14', 'MFCCs_15', 'MFCCs_16', 'MFCCs_17', 'MFCCs_18', 'MFCCs_19', 'MFCCs_20', 'MFCCs_21', 'MFCCs_22']]
    print('[Loader] Select features combination-2.')
    return

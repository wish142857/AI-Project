# -*- coding: utf-8 -*-
import sys
from evaluator import *
from loader import *
from timer import *

FILE_PATH = "../data/train_set.csv"
dataset = None


def load():
    global dataset
    # Load dataset
    try:
        dataset = load_dataset(FILE_PATH)
    except IOError:
        print("[Error] File not found or failed to read!")
        sys.exit()
    except TypeError:
        print("[Error] Format error when loading dataset!")
        sys.exit()
    # Select features
    try:
        # TODO: Select features selected here
        # TODO: select_features_1 | select_features_2
        select_features_1(dataset)
        # select_features_2(dataset)
    except KeyError:
        print("[Error] Key error when selecting features!")


@timer
def run():
    global dataset
    # TODO: Select functions called here
    # TODO: evaluate_naive_bayes | evaluate_decision_tree | evaluate_svm | evaluate_naive_bayes_diy
    print(evaluate_naive_bayes(dataset['data'].values, dataset['target'].values))
    # print(evaluate_decision_tree(dataset['data'].values, dataset['target'].values))
    # print(evaluate_svm(dataset['data'].values, dataset['target'].values))
    # print(evaluate_naive_bayes_diy(dataset['data'].values, dataset['target'].values))


def main():
    load()
    run()


if __name__ == '__main__':
    main()

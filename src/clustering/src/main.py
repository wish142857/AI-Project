# -*- coding: UTF-8 -*-
import sys
from loader import *
from evaluator import *
from painter import *
from timer import *

FILE_PATH = "../data/Frogs_MFCCs.csv"
SAVE_PIC_PATH_1 = '../pic/k_means_result.png'
SAVE_PIC_PATH_2 = '../pic/spectral_clustering_result.png'
SAVE_PIC_PATH_3 = '../pic/k_means_diy_result.png'
dataset = None


def load():
    global dataset
    # Load dataset
    try:
        dataset = load_dataset(FILE_PATH)
    except IOError:
        print('[Error] File not found or failed to read!')
        sys.exit()
    except TypeError:
        print('[Error] Format error when loading dataset!')
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
    # TODO: evaluate_k_means | evaluate_spectral_clustering | evaluate_k_means_diy
    # TODO: paint_k_means | paint_spectral_clustering | paint_k_means_diy
    print(evaluate_k_means(dataset['data'], dataset['target']))
    # print(evaluate_spectral_clustering(dataset['data'], dataset['target']))
    # print(evaluate_k_means_diy(dataset['data'], dataset['target']))
    # print(paint_k_means(dataset['data'], SAVE_PIC_PATH_1))
    # print(paint_spectral_clustering(dataset['data'], SAVE_PIC_PATH_2))
    # print(paint_k_means_diy(dataset['data'], SAVE_PIC_PATH_3))


def main():
    load()
    run()


if __name__ == '__main__':
    main()

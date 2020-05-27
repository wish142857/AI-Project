# -*- coding: utf-8 -*-
from time import time


def timer(fn):
    """ Function to calculate run time
    :param fn {function}
    :return {function}
    """
    def inner():
        start = time()
        fn()
        ret = time() - start
        if ret < 1e-6:
            unit = "ns"
            ret *= 1e9
        elif ret < 1e-3:
            unit = "us"
            ret *= 1e6
        elif ret < 1:
            unit = "ms"
            ret *= 1e3
        else:
            unit = "s"
        print("[Timer] Run time is %.1f %s." % (ret, unit))
    return inner

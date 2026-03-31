from pyeda.inter import *
import numpy as np
import sys
sys.path.append("..")
from BDDset import BDDset

if __name__ == '__main__':

    # construct first BDDset object
    ex = expr("a & b & c & ~(d & e & f)")
    C = np.array([[-1, 0], [0, -1], [1, 1], [0, 1], [1, 0], [-1, -1]])
    d = np.array([[0], [0], [4], [2], [2], [-2]])

    set = BDDset(ex, C, d)

    # compute linear map
    A = np.array([[1, 2], [0, -1]])
    set = set * A

    # visualize the resulting set
    set.plot('b')
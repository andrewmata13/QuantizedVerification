from pyeda.inter import *
import numpy as np
import sys
sys.path.append("..")
from BDDset import BDDset

if __name__ == '__main__':

    # construct first BDDset object
    ex = expr('a & b & c')
    C = np.array([[-1, 0], [0, -1], [1, 1]])
    d = np.array([[0], [0], [4]])

    set1 = BDDset(ex, C, d)

    # construct second BDDset object
    ex = expr('a & b & c & d')
    C = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
    d = np.array([[-1], [-1], [4], [4]])

    set2 = BDDset(ex, C, d)

    # compute set difference
    set = set1.setdiff(set2)

    # visualize the resulting set
    set.plot('y')
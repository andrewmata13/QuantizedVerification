from pyeda.inter import *
import numpy as np
import sys
sys.path.append("..")
from BDDset import BDDset

if __name__ == '__main__':

    # construct first BDDset object
    ex = expr('a & b & c & d')
    C = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    d = np.array([[2], [-1], [4], [-3]])

    set1 = BDDset(ex, C, d)
    set2 = BDDset(ex, C, d)

    # compute set difference
    diff = set1.setdiff(set2)

    print("plotting...")

    # visualize the resulting set
    diff.plot('r')

    print("done")

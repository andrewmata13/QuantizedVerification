from pyeda.inter import *
import numpy as np
import sys
sys.path.append("..")
from BDDset import BDDset

if __name__ == '__main__':

    # construct first BDDset object
    ex = expr("a & b & c & d & ~(e & f & g & h)")
    C = np.array([[-1, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [0, 1]])
    d = np.array([[0], [0], [8], [4], [-5], [-1], [7], [3]])

    set1 = BDDset(ex, C, d)
    set1.plot("yellow", show=False, alpha=0.5)

    # construct second BDDset object
    ex = expr('a & b & c')
    C = np.array([[-1, 0], [0, -1], [1, 1]])
    d = np.array([[0], [0], [9]])

    set2 = BDDset(ex, C, d)
    set2.plot("blue", show=False, alpha=0.5)

    # compute intersection
    set = set1 & set2

    # visualize the resulting set
    set.plot('g', alpha=0.5)

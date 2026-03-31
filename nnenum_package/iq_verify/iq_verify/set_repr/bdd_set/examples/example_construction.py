from pyeda.inter import *
import numpy as np
import sys
sys.path.append("..")
from BDDset import BDDset

if __name__ == '__main__':

    # construct a BDDset object
    ex = expr("a & b & c & d & ~(e & f & g & h)")
    C = np.array([[-1, 0], [0, -1], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [0, 1]])
    d = np.array([[0], [0], [8], [4], [-5], [-1], [7], [3]])

    set = BDDset(ex, C, d)

    set.bdd_to_png('out.png')

    # visualize the set
    set.plot('r')

from pyeda.inter import *
import numpy as np
import sys
sys.path.append("..")
from BDDset import BDDset
from sets.Polytope import Polytope

if __name__ == '__main__':

    # construct first BDDset object
    ex = expr("a & b & c")
    C = np.array([[-1, 0], [0, -1], [1, 1]])
    d = np.array([[0], [0], [3]])

    set1 = BDDset(ex, C, d)

    # construct second BDDset object
    ex = expr("a & b & c")
    C = np.array([[-1, 0], [0, -1], [1, 1]])
    d = np.array([[-1], [-1], [4]])

    set2 = BDDset(ex, C, d)

    # compute intersection
    u = set1 | set2

    print(f"union varnames: {u.var_list}")

    # visualize the resulting set
    u.plot('g')

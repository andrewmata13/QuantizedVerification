from pyeda.inter import *
import numpy as np
import sys
sys.path.append("..")
from BDDset import BDDset
from sets.Polytope import Polytope
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # construct BDDset object
    ex = expr("a & b & c")
    C = np.array([[-1, 0], [0, -1], [1, 1]])
    d = np.array([[0], [0], [3]])

    set = BDDset(ex, C, d)

    # check if pt within set
    pt = np.array([0.5, 0.2])
    contains = set.contains(pt)
    plt.scatter(x=pt[0], y=pt[1], c='r', marker='x' if contains else 'o', zorder=2)

    # check if pt within set
    pt = np.array([2.0, 2.0])
    contains = set.contains(pt)
    plt.scatter(x=pt[0], y=pt[1], c='r', marker='x' if contains else 'o', zorder=2)

    # visualize the resulting set
    set.plot('g')

    ### More in-depth example
    pts = np.array([
        [0.5, 0.2],
        [2.0, 2.0],
        [-0.3, 0.7],
        [0.2, -1.1],
        [1.5, 1.0]
        ])
    for pt in pts:
        contains = set.contains(pt)
        plt.scatter(x=pt[0], y=pt[1], c='r', marker='x' if contains else 'o', zorder=2)
    set.plot('g')

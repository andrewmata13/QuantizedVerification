import pypoman
from pyeda.inter import *
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

class Polytope:
    """class representing a polytope in halfspace representation P = {x | C * x <= d}"""

    def __init__(self, C, d):
        """class constructor"""

        self.C = C                          # matrix C for the inequality constraint C*x <= d
        self.d = d                          # constant offset for the inequality constraint C*x <= d

    def vertices(self):
        """compute the vertices of a polytope"""
        return pypoman.compute_polytope_vertices(self.C, self.d)

    def plot(self, color):
        """plot the polytope"""

        v = self.vertices()
        v.append(v[0])
        v_ = np.resize(v[0], (2, 1))
        for i in range(1, len(v)):
            v_ = np.concatenate((v_, np.resize(v[i], (2, 1))), axis=1)
        hull = ConvexHull(np.transpose(v_))
        plt.fill(v_[0, hull.vertices], v_[1, hull.vertices], color)
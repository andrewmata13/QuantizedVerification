import numpy as np
import matplotlib.pyplot as plt
import copy

DIMS_TO_PLOT = (0, 1)

import iq_verify.set_repr
from iq_verify.set_repr import set_repr, bounding_box, h_polytope, star_set

class Point(set_repr.SetRepr):
    def __init__(self, n_dims, coords, transformations=None, discrete=None):
        super().__init__(n_dims, discrete)
        self.coords = coords
        self.transformations = transformations
        if self.transformations is None:
            self.transformations = []

        # Shortcut for undo_all_transformations: keep a copy of the Point at time
        # of construction. Unlike H-Polytopes, which may undergo a combination of
        # both transformations and additional constraints, a Point is determined
        # solely by its coordinates, which we can easily reset.
        self.original_params = (n_dims, coords.copy(), copy.deepcopy(transformations), copy.deepcopy(discrete))

    def plot(self, color, fill=True):
        # TODO
        self.plot_2D(color, fill=True)

    def plot_2D(self, color, dims=None, fill=True):
        if dims is None:
            global DIMS_TO_PLOT
            dims = DIMS_TO_PLOT
        plt.plot(self.coords[dims[0]], self.coords[dims[1]], color=color, marker=".")

    def maximize(self, dir_vec):
        return self.coords

    def get_verts(self):
        return self.coords

    def affine_transform(self, mat, offset, in_place=False):
        # Reshape because we require matrix @ matrix multiplication
        new_coords = mat @ self.coords.reshape((self.n_dims, 1)) + offset
        rv = self if in_place else self.copy()
        rv.coords = new_coords
        rv.transformations.append((mat, offset))
        return rv

    def undo_all_transformations(self):
        # TODO: we could provide "discrete" as a keyword argument and override rv's discrete dict,
        # if it must be different from the original discrete dict
        rv = Point(*self.original_params)
        return rv

    def split_on_halfspace_constraint(self, lhs, rhs):
        if lhs @ self.coords > rhs:
            rv = (self.copy(), None)
        else:
            rv = (None, self.copy())
        return rv

    def get_bounding_box(self):
        ivals = []
        for d in range(self.n_dims):
            ivals.append([float(self.coords[d]), float(self.coords[d])])
        box = bounding_box.BoundingBox(self.n_dims, ivals)
        return box

    def __str__(self):
        return "Point: " + str(self.coords) + " " + str(self.properties)

    def copy(self):
        rv = Point(
            n_dims=self.n_dims,
            coords=self.coords.copy(),
            transformations=copy.deepcopy(self.transformations),
            discrete=copy.deepcopy(self.properties),
        )
        return rv

    def check_is_feasible(self):
        return True

    def check_for_intersection(self, other):
        return self.intersect(other)

    def intersect(self, other):
        assert isinstance(other, h_polytope.HPoly) or isinstance(other, bounding_box.BoundingBox) or isinstance(other, star_set.StarSet), f"Intersection of Point and {type(other)} not implemented"
        if other.contains(self.coords):
            rv = self.copy()
        else:
            rv = None
        return rv

    def set_difference(self, other):
        assert isinstance(other, h_polytope.HPoly) or isinstance(other, bounding_box.BoundingBox), f"Set difference of Point and {type(other)} not implemented"
        if other.contains(self.coords):
            rv = []
        else:
            rv = [self.copy()]
        return rv

    def get_chebyshev_center(self):
        # (radius, center)
        return 0, self.coords.copy()

import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import itertools
import copy
import logging
log = logging.getLogger(__name__)
import cvxpy

import iq_verify.set_repr
from iq_verify.set_repr import set_repr, bounding_box, h_polytope, v_polytope

DIMS_TO_PLOT = (0, 1)

# TODO: Not a SetRepr because it doesn't do intersections or splitting.
class Zonotope:
    def __init__(self,
        n_dims,
        c,
        g_mat,
        transformations=None,
        discrete=None,
    ):
        # The number of dimensions this stateset occupies in euclidean space.
        self.n_dims = n_dims

        # A dictionary containing values for any other discrete variables used
        # by the stateset in reachability or simulation.
        self.properties = discrete
        if self.properties is None:
            self.properties = {}

        self.c = c
        assert isinstance(self.c, np.ndarray) and self.c.shape == (self.n_dims, 1), f"Expected center to be an Nx1 matrix, got {self.c.shape}"

        self.g_mat = g_mat
        assert isinstance(self.g_mat, np.ndarray) and self.g_mat.shape[0] == self.n_dims, f"Expected g_mat to be a matrix with {self.n_dims} rows, got {self.g_mat.shape}"

        # Track the affine transformations (A, b) that have been performed
        # on this HPoly, in case we need to "undo" them later. For example,
        # when forward reachability reveals an intersection with the unsafe set,
        # we may wish to trace the intersection back to its original location.
        self.transformations = transformations
        if self.transformations is None:
            self.transformations = []

    def plot(self, color, alpha=1.0, fill=False):
        if self.n_dims != 2:
            log.debug(f"Cannot plot Zonotope with {self.n_dims} dimensions")
            return

        self.plot_2D(color, alpha, fill)

    def plot_2D(self, color, alpha=1.0, fill=False):
        """Plot the set"""
        verts = self.get_verts()
        vpoly = v_polytope.VPoly(n_dims=self.n_dims, verts=verts)
        vpoly = vpoly.project_to_2D(i=DIMS_TO_PLOT[0], j=DIMS_TO_PLOT[1])
        vpoly.plot(color=color, fill=fill, alpha=alpha)

    def maximize(self, dir_vec):
        """Return the set's extreme point in the given direction"""
        dir_vec = dir_vec.flatten()
        max_vert = None
        for v in self.get_verts():
            v = v.flatten()
            if (max_vert is None) or (np.dot(v, dir_vec) > np.dot(max_vert, dir_vec)):
                max_vert = v
        rv = max_vert.reshape(self.c.shape)
        return rv

    def get_verts(self):
        """Return the vertices of the set's convex hull"""
        vert_generators = [np.array(p).reshape(self.n_dims, 1) for p in itertools.product([1.0, -1.0], repeat=self.g_mat.shape[1])]
        rv = [(self.c + (self.g_mat @ vg)) for vg in vert_generators]
        return rv

    def affine_transform(self, A, b, in_place=False):
        """Return this set after the affine transformation"""
        rv = self if in_place else self.copy()

        rv.g_mat = rv.g_mat @ A
        rv.c = (A @ rv.c) + b
        rv.transformations = self.transformations + [(A, b)]

        return rv

    def split_on_halfspace_constraint(self, lhs, rhs):
        """Return the (greater_than, less_than) portions of this stateset with respect to the constraint"""
        return self.copy(), self.copy()
        # TODO: Another idea would be to convert this Zono to a StarSet,
        # add the constraint to the StarSet creating star1 and star2,
        # overapproximate star1 and star2 via BoundingBox into box1 and box2,
        # then initialize zono1 and zono2 using the axis-aligned constraints.
        #
        # Could be slightly fancier and attempt to use PCA directions for non-axis-aligned constraints.

    def get_bounding_box(self):
        """Returning a bounding box that contains this stateset"""
        verts = self.get_verts()
        intervals = []
        for i in range(self.n_dims):
            lo = min([v[i, 0] for v in verts])
            hi = max([v[i, 0] for v in verts])
            intervals.append([lo, hi])

        rv = bounding_box.BoundingBox(n_dims=self.n_dims, intervals=intervals)
        return rv

    def check_is_feasible(self):
        """Return True iff this set is "feasible," e.g. has satisfiable constraints."""
        return True

    def __str__(self):
        """Return a string representation of this set"""
        rv = f"Zonotope:\nc: {self.c}\ng_mat: {self.g_mat}"
        return rv

    def copy(self):
        """Return a copy of this set"""
        rv = Zonotope(
            n_dims=self.n_dims,
            c=copy.deepcopy(self.c),
            g_mat=copy.deepcopy(self.g_mat),
            transformations=copy.deepcopy(self.transformations),
            discrete=copy.deepcopy(self.properties),
        )
        return rv

    def check_for_intersection(self, other):
        """Return True if and only if the two sets intersect"""
        if not (
                isinstance(other, bounding_box.BoundingBox) or \
                isinstance(other, h_polytope.HPoly) or \
                isinstance(other, Zonotope)
        ):
            raise ValueError(f"Zonotope intersection checking not implemented with another set of type {type(other)}")

        """
        Shortcut: Check Zono's bounding box as coarse overapproximation
        """
        shortcut_zono_bbox = True
        if shortcut_zono_bbox and isinstance(other, bounding_box.BoundingBox):
            # If even the box overapproximation of this zonotope doesn't intersect the other set,
            # then there is no need to check for exact intersection.
            zono_box = self.get_bounding_box()
            if zono_box.check_for_intersection(other) == False:
                return False

        """
        Shortcut: Attempt to find a separating hyperplane
        """
        # Let this zonotope be Z, and the other set be U.
        dir_Z_to_U = (other.get_chebyshev_center()[1]).flatten() - (self.get_chebyshev_center()[1]).flatten()
        dir_U_to_Z = -1 * dir_Z_to_U

        vZ = np.array(self.maximize(dir_Z_to_U))
        vU = np.array(other.maximize(dir_U_to_Z))

        if np.dot(vU.flatten(), dir_Z_to_U) > np.dot(vZ.flatten(), dir_Z_to_U):
            # The dir vector defines a hyperplane (orthogonal to dir vec) which separates Z from U.
            # Therefore, they do not intersect.
            return False

        """
        None of the shortcuts were conclusive, so we must perform an LP to check precisely for intersection
        """
        # TODO: Since we are only using Zonotope as an outer-approximation for StarSet,
        # we can rely on the StarSet's LP. There is no need to actually encode
        # the LP for Zonotopes. For now, we can still perform sound verification by
        # simply returning "Yes, this zonotope intersects other".
        return True

        # TODO: A more complete approach would implement the following:
        #
        # Formulate LP: Can we find a linear combination of the Zonotope's
        # vertices that resides within the H-Polytope's halfspace constraints.


    def intersect(self, other):
        """TODO"""
        return self.copy()

    def set_difference(self, other):
        """Return this set minus other set; if there is no intersection, return this set"""
        return self.copy()

    def undo_all_transformations(self):
        """TODO"""
        rv = self.copy()
        for (A, b) in reversed(self.transformations):
            try:
                A_inv = np.linalg.inv(A)
            except numpy.linalg.LinAlgError:
                raise ValueError(f"Transformation matrix {A} is not invertible")

            # Perform the opposite of the original affine transformation.
            reverted_g_mat = rv.g_mat @ A_inv
            reverted_c = A_inv @ (rv.c - b)

            rv = Zonotope(
                n_dims=self.n_dims,
                c=reverted_c,
                g_mat=reverted_g_mat,
                transformations=None,
            )

        return rv

    def get_chebyshev_center(self):
        """return radius, center"""
        return None, self.c

    @staticmethod
    def from_BoundingBox(bbox):
        n_dims = bbox.n_dims
        _, c = bbox.get_chebyshev_center()
        c = c.reshape((n_dims,1))
        g_mat = np.zeros(shape=(n_dims,0))
        for dim in range(n_dims):
            r = 0.5 * abs(bbox.u(dim) - bbox.l(dim))
            col = np.zeros((n_dims, 1))
            col[dim, 0] = r
            g_mat = np.hstack((g_mat, col))

        rv = Zonotope(
            n_dims=n_dims,
            c=c,
            g_mat=g_mat,
        )

        return rv

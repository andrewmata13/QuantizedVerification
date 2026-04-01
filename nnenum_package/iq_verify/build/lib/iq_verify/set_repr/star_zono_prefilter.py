import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import itertools
import copy
import logging
log = logging.getLogger(__name__)
import cvxpy

import iq_verify.set_repr
from iq_verify.set_repr import set_repr, bounding_box, h_polytope, v_polytope, star_set, zonotope

DIMS_TO_PLOT = (0, 1)

class StarZonoPrefilter(set_repr.SetRepr):
    def __init__(self,
        n_dims,
        zono=None,
        star=None,
        transformations=None,
        discrete=None,
    ):

        super().__init__(n_dims, discrete)

        self.zono = zono
        self.star = star

        if self.star is None:
            assert self.zono is not None

            unit_square = h_polytope.HPoly(
                n_dims=self.n_dims,
                LHS=np.vstack((
                    np.eye(self.n_dims),
                    -np.eye(self.n_dims),
                )),
                rhs=np.ones((2*self.n_dims, )),
            )

            self.star = star_set.StarSet(
                n_dims=self.n_dims,
                c=self.zono.c,
                g_mat=self.zono.g_mat,
                hpoly=unit_square,
                discrete=discrete,
            )

        if self.zono is None:
            assert self.star is not None
            self.zono = zonotope.Zonotope.from_BoundingBox(self.star.get_bounding_box())

        # Track the affine transformations (A, b) that have been performed
        # on this HPoly, in case we need to "undo" them later. For example,
        # when forward reachability reveals an intersection with the unsafe set,
        # we may wish to trace the intersection back to its original location.
        self.transformations = transformations
        if self.transformations is None:
            self.transformations = []


    def plot(self, color, fill=False, alpha=1.0):
        """Plot the set"""
        #self.zono.plot("red", fill=fill, alpha=alpha)
        self.star.plot(color, fill=fill, alpha=alpha)

    def maximize(self, dir_vec):
        """Return the set's extreme point in the given direction"""
        return self.zono.maximize(dir_vec)

    def get_verts(self):
        """Return the vertices of the set's convex hull"""
        return self.zono.get_verts()

    def affine_transform(self, A, b, in_place=False):
        """Return this set after the affine transformation"""
        rv = self if in_place else self.copy()

        rv.zono = rv.zono.affine_transform(A, b, in_place=in_place)
        rv.star = rv.star.affine_transform(A, b, in_place=in_place)
        rv.transformations += [(A, b)]

        return rv

    def split_on_halfspace_constraint(self, lhs, rhs):
        """Return the (greater_than, less_than) portions of this stateset with respect to the constraint"""
        star2_gt, star2_lt = self.star.split_on_halfspace_constraint(lhs, rhs)

        rv_gt = self.copy()
        rv_gt.star = star2_gt

        rv_lt = self.copy()
        rv_lt.star = star2_lt

        return rv_gt, rv_lt

    def get_bounding_box(self):
        """Returning a bounding box that contains this stateset"""
        return self.star.get_bounding_box()

    def check_is_feasible(self):
        """Return True iff this set is "feasible," e.g. has satisfiable constraints."""
        return self.star.check_is_feasible()

    def __str__(self):
        """Return a string representation of this set"""
        return f"StarZonoPrefilter:\n{self.star.__str__()}\n{self.zono.__str__()}"

    def copy(self):
        """Return a copy of this set"""
        rv = StarZonoPrefilter(
            n_dims=self.n_dims,
            zono=self.zono.copy(),
            star=self.star.copy(),
            transformations=copy.deepcopy(self.transformations),
            discrete=copy.deepcopy(self.properties),
        )
        return rv

    def check_for_intersection(self, other):
        """Return True if and only if the two sets intersect"""
        if self.zono.check_for_intersection(other) == False:
            # If the zono overapproximation doesn't even intersect with other,
            # then the underlying star definitely doesn't intersect, either.
            return False

        # The zonotope intersection check was inconclusive. Use LP to check
        # whether the star intersects with other.
        rv = self.star.check_for_intersection(other)

        reset_spurious_zono = False
        if rv == False and (reset_spurious_zono == True):
            # The zonotope overapproximation was too conservative, leading the
            # zonotope to intersect "other" while the underlying star does not.
            #
            # Reset the zonotope to be the box bound of the star, in hopes that
            # this gives a tighter overapproximation.
            self.zono = zonotope.Zonotope.from_BoundingBox(self.star.get_bounding_box())
            #
            # TODO: Zonotope Domain Contraction: https://par.nsf.gov/servlets/purl/10419676

        return rv

    def intersect(self, other):
        rv_zono = self.zono.intersect(other)
        rv_star = self.star.intersect(other)
        rv = StarZonoPrefilter(
            n_dims=self.n_dims,
            zono=rv_zono,
            star=rv_star,
            transformations=copy.deepcopy(self.transformations),
            discrete=copy.deepcopy(self.properties),
        )
        return rv

    def set_difference(self, other):
        """Return this set minus other set; if there is no intersection, return this set"""
        if not self.check_for_intersection(other):
            return [self]

        rv = []
        for star in self.star.set_difference(other):
            rv.append(StarZonoPrefilter(
                n_dims=self.n_dims,
                star=star,
                transformations=copy.deepcopy(self.transformations),
                discrete=copy.deepcopy(self.properties),
            ))
        return rv


    def undo_all_transformations(self):
        rv_zono = self.zono.undo_all_transformations()
        rv_star = self.star.undo_all_transformations()
        rv = StarZonoPrefilter(
            n_dims=self.n_dims,
            zono=rv_zono,
            star=rv_star,
            transformations=copy.deepcopy(self.transformations),
            discrete=copy.deepcopy(self.properties),
        )
        return rv

    def get_chebyshev_center(self):
        """return radius, center"""
        return self.star.get_chebyshev_center()

    @staticmethod
    def from_BoundingBox(bbox):
        zono = zonotope.Zonotope.from_BoundingBox(bbox)
        rv = StarZonoPrefilter(
            n_dims=bbox.n_dims,
            zono=zono,
        )
        return rv

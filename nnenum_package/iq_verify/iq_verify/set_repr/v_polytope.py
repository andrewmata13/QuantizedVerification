import numpy as np
import matplotlib.pyplot as plt
import logging
log = logging.getLogger(__name__)
import copy

import iq_verify.set_repr.bounding_box

class VPoly:
    """
    V-Polytope, used mostly for plotting.
    Does not fully conform to the set_repr abstract base class.
    """
    def __init__(self, n_dims, verts):
        self.n_dims = n_dims

        # verts should be a list/array of lists/arrays of length n_dims
        self.verts = verts
        for v in self.verts:
            assert len(v.flatten()) == self.n_dims, f"Vertex {v} invalid for n_dims={self.n_dims}"

    def project_to_2D(self, i=0, j=1):
        """
        Extract only two of the dimensions (configurable) from this VPoly's vertices
        """
        if self.n_dims == 2:
            return self.copy()

        verts_2D = []
        for v in self.verts:
            vert_2D = np.array([v[i], v[j]])
            verts_2D.append(vert_2D)
        rv = VPoly(n_dims=2, verts=verts_2D)
        return rv

    def plot(self, color, fill=False, alpha=1.0):
        """Plot the set"""
        if not self.check_is_feasible():
            return

        if self.n_dims == 2:
            self.plot_2D(color, fill, alpha)
        else:
            raise ValueError(f"Plotting not implemented in {self.n_dims} dimensions")

    def plot_2D(self, color, fill, alpha):
        sorted_verts = []
        for angle in np.linspace(0, 2 * np.pi, 100):
            x = np.cos(angle)
            y = np.sin(angle)
            dir_vec = np.array([
                [x],
                [y],
            ])
            v = self.maximize(dir_vec)
            sorted_verts.append(v)
        sorted_verts.append(sorted_verts[0])
        plt.plot(
            [v[0] for v in sorted_verts],
            [v[1] for v in sorted_verts],
            color,
            marker=",",
        )
        if fill:
            plt.fill(
                [v[0] for v in sorted_verts],
                [v[1] for v in sorted_verts],
                color,
                alpha=alpha,
            )

    def maximize(self, dir_vec):
        """Return the set's extreme point in the given direction"""
        max_vert = None
        max_dot = None
        for v in self.verts:
            dot = np.dot(v.flatten(), dir_vec.flatten())
            if max_vert is None or dot > max_dot:
                max_vert = v
                max_dot = dot
        return max_vert

    def get_verts(self):
        """Return the vertices of the set's convex hull"""
        return self.verts

    def affine_transform(self, A, b, in_place=False):
        """Return this set after the affine transformation"""
        new_verts = []
        for v in self.verts:
            new_vert = b + np.dot(A, v)
            new_verts.append(new_vert)
        rv = VPoly(n_dims=self.n_dims, verts=new_verts)
        return rv

    def split_on_halfspace_constraint(self, lhs, rhs):
        """Return the (greater_than, less_than) portions of this stateset with respect to the constraint"""
        left_verts = []
        right_verts = []
        for v in self.verts:
            if np.dot(v, lhs) <= rhs:
                # is left
                left_verts.append(v)
            else:
                right_verts.append(v)

        left_rv = VPoly(n_dims=self.n_dims, verts=left_verts)
        right_rv = VPoly(n_dims=self.n_dims, verts=right_verts)
        return (right_rv, left_rv)

    def get_bounding_box(self):
        """Returning a bounding box that contains this stateset"""
        intervals = [
            [None, None] for _ in range(self.n_dims)
        ]
        for v in self.verts:
            for d in range(self.n_dims):
                min_so_far = intervals[d][0]
                if min_so_far is None or v[d] < min_so_far:
                    intervals[d][0] = v[d]

                max_so_far = intervals[d][1]
                if max_so_far is None or v[d] > max_so_far:
                    intervals[d][1] = v[d]

        rv = bounding_box.BoundingBox(n_dims=self.n_dims, intervals=intervals)
        return rv

    def __str__(self):
        """Return a string representation of this set"""
        rv = "Vpoly: [\n"
        for v in self.verts:
            rv += "\t"
            rv += str(v)
            rv += "\n"
        rv += "]"
        return rv

    def copy(self):
        """Return a copy of this set"""
        rv = VPoly(n_dims=self.n_dims, verts=copy.deepcopy(self.verts))
        return rv

    def check_is_feasible(self):
        rv = len(self.verts) > self.n_dims
        return rv

import numpy as np
import matplotlib.pyplot as plt
import itertools

DIMS_TO_PLOT = (0, 1)

class BoundingBox:
    def __init__(self, n_dims, intervals, discrete=None):
        self.n_dims = n_dims

        # intervals is Nx2 list of ranges:
        # [[ x_min, x_max ]
        #  [ y_min, y_max ]
        #  [ ..., ... ]]
        self.properties = discrete
        try:
            self.intervals = np.array(intervals)
            assert self.intervals.shape == (self.n_dims, 2)
        except:
            raise ValueError(f"BoundingBox input expected to be {self.n_dims}x2 array, instead got {self.intervals.shape}")

    def get_verts(self):
        """
        Return the hyperrectangle's vertices as a list
        """
        verts = [*(itertools.product(*self.intervals))]
        return verts

    def maximize(self, dir_vec):
        """Return the set's extreme point in the given direction"""
        max_vert = None
        for v in self.get_verts():
            if (max_vert is None) or (np.dot(v, dir_vec) > np.dot(max_vert, dir_vec)):
                max_vert = v
        return max_vert

    def plot(self, color, fill=False, alpha=1.0):
        """
        Plots a 2D projection of this interval.
        """
        if self.n_dims == 2:
            box_2D = self
        else:
            global DIMS_TO_PLOT

            if DIMS_TO_PLOT is None:
                box_2D = BoundingBox(n_dims=2, intervals=self.intervals[:2])
            else:
                first = DIMS_TO_PLOT[0]
                second = DIMS_TO_PLOT[1]

                box_2D = BoundingBox(
                    n_dims=2,
                    intervals=[
                        self.intervals[first],
                        self.intervals[second],
                    ],
                )

        box_2D.plot_2D(color, fill, alpha)

    def plot_2D(self, color, fill=False, alpha=1.0):
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

    def l(self, dim):
        """
        Returns the lower bound in the given dimension
        """
        return self.intervals[dim][0]

    def u(self, dim):
        """
        Returns the upper bound in the given dimension
        """
        return self.intervals[dim][1]

    def __str__(self):
        return str(self.intervals)

    def copy(self):
        return BoundingBox(
            n_dims=self.n_dims,
            intervals=self.intervals.copy(),
        )

    def check_for_intersection(self, other):
        """
        Check whether this interval intersects another interval. Can be done without LP.
        """
        assert self.n_dims == other.n_dims

        assert isinstance(other, BoundingBox), f"Intersection of BoundingBox with {type(other)} not implemented"

        for dim in range(self.n_dims):
            if self.l(dim) > other.u(dim) or self.u(dim) < other.l(dim):
                # In at least one dimension, this box is entirely on one side of the other box
                return False

        # There is some overlap in every dimension, so the boxes intersect.
        return True

    def contains_bounding_box(self, other):
        rv = True
        assert other.n_dims == self.n_dims
        for d in range(self.n_dims):
            rv = rv and (self.intervals[d][0] < other.intervals[d][0])
            rv = rv and (self.intervals[d][1] > other.intervals[d][1])
        return rv

    def get_chebyshev_center(self):
        """return radius, center"""
        center = np.zeros(self.n_dims)
        radii = np.zeros(self.n_dims)

        for dim in range(self.n_dims):
            r = 0.5 * abs(self.u(dim) - self.l(dim))
            radii[dim] = r
            center[dim] = self.l(dim) + r

        return min(radii), center

    def contains(self, pt):
        """
        Returns True iff this interval contains the given point
        """
        assert isinstance(pt, np.ndarray)
        pt = pt.flatten()
        assert len(pt) == self.n_dims

        for dim in range(self.n_dims):
            if pt[dim] < self.l(dim) or pt[dim] > self.u(dim):
                return False # the pt is outside the interval in this dimension

        return True

    def get_constraints(self):
        """
        Returns the halfspace constraints corresponding to this hyperrectangle
        """
        LHS = []
        rhs = []
        for n in range(self.n_dims):
            LHS_lo = [0] * self.n_dims
            LHS_lo[n] = -1.0
            rhs_lo = -1 * self.intervals[n][0]
            LHS.append(LHS_lo)
            rhs.append(rhs_lo)

            LHS_hi = [0] * self.n_dims
            LHS_hi[n] = 1.0
            rhs_hi = self.intervals[n][1]
            LHS.append(LHS_hi)
            rhs.append(rhs_hi)

        return np.array(LHS), np.array(rhs)

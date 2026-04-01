import numpy as np
import scipy
import copy
import itertools

from iq_verify.set_repr import h_polytope, bounding_box, star_set

class AggregationUtils:
    def __init__(self):
        pass

    @staticmethod
    def aggregate_into_box_BoundingBox(Ts, n_dims):
        intervals = None
        for T in Ts:
            bbox = T.get_bounding_box()
            if intervals is None:
                intervals = bbox.intervals
            else:
                for n in range(bbox.n_dims):
                    intervals[n][0] = min(bbox.intervals[n][0], intervals[n][0])
                    intervals[n][1] = max(bbox.intervals[n][1], intervals[n][1])

        rv_box = bounding_box.BoundingBox(
            n_dims=len(intervals),
            intervals=intervals,
            discrete=copy.deepcopy(Ts[0].properties),
        )
        return rv_box

    @staticmethod
    def aggregate_into_box_HPoly(Ts, n_dims):
        bbox = AggregationUtils.aggregate_into_box_BoundingBox(Ts, n_dims)
        rv = h_polytope.HPoly.from_BoundingBox(bbox)
        return rv

    @staticmethod
    def aggregate_into_box_StarSet(Ts, n_dims):
        hpoly = AggregationUtils.aggregate_into_box_HPoly(Ts, n_dims)
        rv = star_set.StarSet.from_HPoly(hpoly)
        return rv

    @staticmethod
    def aggregate_into_pca_HPoly(Ts, n_dims):
        if len(Ts) == 1 and isinstance(Ts[0], h_polytope.HPoly):
            return Ts[0]

        rv_LHS = []
        rv_rhs = []

        # Find the max/min directions of variance for the sets
        points = []
        for T in Ts:
            _, c = T.get_chebyshev_center()
            points.append(c)

        pts_arr = np.array(points).reshape(-1, n_dims)
        U, s, Vh = scipy.linalg.svd(pts_arr, full_matrices=True)
        pca_dirs = [v for v in Vh]

        # As a short-term, hacky heuristic: Maximize in axis-aligned directions as well.
        HACK_axis_aligned_dirs = True
        if HACK_axis_aligned_dirs:
            for i in range(n_dims):
                lo_dir = np.zeros((n_dims, ))
                lo_dir[i] = -1.0
                pca_dirs.append(lo_dir)

                hi_dir = np.zeros((n_dims))
                hi_dir[i] = 1.0
                pca_dirs.append(hi_dir)

        # As a short-term, hacky heuristic: Maximize in the directions of each corner.
        HACK_corners_dirs = True
        if HACK_corners_dirs:
            pca_dirs += [np.array(p) for p in itertools.product([-1.0, 1.0], repeat=n_dims)]

        # NOT recommended!
        # As a short-term, hacky heuristic: Maximize in additional random
        # directions as a supplement to the 2n directions given by PCA. As the
        # number of random directions increases, the aggregation approximates
        # the convex hull.
        HACK_random_dirs = False
        if HACK_random_dirs:
            N_RAND_DIRS = 100
            for _ in range(N_RAND_DIRS):
                random_dir = np.random.default_rng().uniform(-1, 1, n_dims)
                pca_dirs.append(random_dir)

        for pca_dir in pca_dirs:
            M = float(max([pca_dir @ T.maximize(pca_dir) for T in Ts]))
            m = float(max([-pca_dir @ T.maximize(-pca_dir) for T in Ts]))

            rv_LHS.append(pca_dir)
            rv_rhs.append(M)
            rv_LHS.append(-pca_dir)
            rv_rhs.append(m)

        rv = h_polytope.HPoly(
            n_dims=n_dims,
            LHS=np.array(rv_LHS),
            rhs=np.array(rv_rhs),
            discrete=copy.deepcopy(Ts[0].properties),
        )
        return rv

    @staticmethod
    def aggregate_into_pca_StarSet(Ts, n_dims):
        if len(Ts) == 1 and isinstance(Ts[0], star_set.StarSet):
            return Ts[0]

        hpoly = AggregationUtils.aggregate_into_pca_HPoly(Ts, n_dims)
        rv = star_set.StarSet.from_HPoly(hpoly)
        return rv

import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import itertools
import copy
import logging
log = logging.getLogger(__name__)
import cvxpy
LP_SOLVER = cvxpy.MOSEK
#LP_SOLVER = cvxpy.GLPK
#LP_SOLVER = cvxpy.GLPK_MI  # NOTE: GLPK_MI failed miserably at finding the Chebyshev center of a StarSet
#LP_SOLVER = cvxpy.GUROBI
#LP_SOLVER = cvxpy.CVXOPT

import iq_verify.set_repr
from iq_verify.set_repr import set_repr, bounding_box, h_polytope, v_polytope

N_PLOTTING_DIRS = 100
DIMS_TO_PLOT = (0, 1)

class StarSet(set_repr.SetRepr):
    def __init__(self,
        n_dims,
        c,
        g_mat,
        hpoly,
        discrete=None,
        original_c=None,
        original_g_mat=None,
        transformations=None,
    ):
        super().__init__(n_dims, discrete)
        self.c = c
        assert isinstance(self.c, np.ndarray) and self.c.shape == (self.n_dims, 1), f"Expected center to be an Nx1 matrix, got {self.c.shape}"
        self.g_mat = g_mat
        assert isinstance(self.g_mat, np.ndarray) and self.g_mat.shape[0] == self.n_dims, f"Expected g_mat to be a matrix with {self.n_dims} rows, got {self.g_mat.shape}"

        # All constraints are of the form "x1 + x2 + ... + xN <= z"
        # Therefore, constr_mat should be a _xN matrix, and
        # constr_rhs should be a vector of length N.
        self.hpoly = hpoly

        # In case we need to revert this StarSet after performing transformations.
        self.original_c = original_c
        if original_c is None:
            self.original_c = c.copy()
        self.original_g_mat = original_g_mat
        if original_g_mat is None:
            self.original_g_mat = g_mat.copy()

        # Track the affine transformations (A, b) that have been performed
        self.transformations = transformations
        if self.transformations is None:
            self.transformations = []

    def affine_transform(self, mat, offset, in_place=False):
        """
        Return the star set obtained by performing the given affine
        transformation on this star set.
        """
        assert isinstance(mat, np.ndarray) and mat.shape == (self.n_dims, self.n_dims), f"Expecting A_mat to be an NxN matrix, got {mat.shape}"
        assert isinstance(offset, np.ndarray) and offset.shape == (self.n_dims, 1), f"Expecting offset to be an Nx1 matrix, got {offset.shape}"

        # NOTE: Do not deepcopy the hpoly here. The copy will be made by the called function.
        rv = self if in_place else self.copy(hpoly=self.hpoly)

        rv.c = mat @ rv.c + offset
        rv.g_mat = mat @ rv.g_mat
        rv.transformations += [(mat, offset)]
        return rv

    def maximize(self, ran_dir):
        """Return the set's extreme point in the given direction"""

        # convert to domain direction
        dom_dir = np.dot(ran_dir.T, self.g_mat)

        # maximize in domain direction
        dom_vert = self.hpoly.maximize(dom_dir)

        # convert to point in range
        rv = np.dot(self.g_mat, dom_vert) + self.c
        return rv

    def get_verts(self):
        """Return the vertices of the set's convex hull"""
        # TODO: This only samples some directions; it could potentially miss a
        # vertex, especially in higher dimensions.
        verts = []
        '''
        for angle in np.linspace(0, 2 * np.pi, N_PLOTTING_DIRS):
            # get range direction from angle
            x = np.cos(angle)
            y = np.sin(angle)
            dir_vec = np.array([
                [x],
                [y],
            ])
            vert = self.maximize(dir_vec)
            verts.append(vert)
        '''
        for _ in range(N_PLOTTING_DIRS):
            rand_dir = np.random.default_rng().uniform(low=-1, high=1.0, size=(self.n_dims, 1))
            vert = self.maximize(rand_dir)
            verts.append(vert)
        return verts

    def check_is_feasible(self):
        rv = self.hpoly.check_is_feasible()
        return rv

    def plot(self, color, alpha=1.0, fill=False):
        """Plot the StarSet"""
        if self.check_is_feasible() == False:
            log.warn("Cannot plot infeasible StarSet")
            return

        try:
            self.plot_2D(color, alpha, fill)
        except h_polytope.InfeasibleException:
            return

    def plot_2D(self, color, alpha, fill=False):
        """Plot in two dimensions"""
        verts = self.get_verts()
        vpoly = v_polytope.VPoly(n_dims=self.n_dims, verts=verts)
        vpoly = vpoly.project_to_2D(i=DIMS_TO_PLOT[0], j=DIMS_TO_PLOT[1])
        vpoly.plot(color=color, fill=fill, alpha=alpha)


    def __str__(self):
        rv = f"c:\n{self.c}\nG:\n{self.g_mat}\nHPoly:\n{self.hpoly}"
        return rv

    def copy(self, hpoly=None):
        if hpoly is None:
            hpoly = self.hpoly.copy()
        """Return a copy of this StarSet"""
        rv = StarSet(
            n_dims=self.n_dims,
            c=self.c.copy(),
            g_mat=self.g_mat.copy(),
            hpoly=hpoly,
            original_c=self.original_c.copy(),
            original_g_mat=self.original_g_mat.copy(),
            discrete=copy.deepcopy(self.properties),
            transformations=copy.deepcopy(self.transformations),
        )
        return rv

    def split_on_halfspace_constraint(self, ran_lhs, ran_rhs):
        # 5) Convert the halfspace "x < 0" from a constraint in the range to
        #    a constraint in the domain.
        #          hx <= f    ---->    hGa <= f - hc
        #          1x <= 0    ---->    Ga <= 0 - c
        d_lhs_arr = ran_lhs @ self.g_mat
        d_rhs_arr = ran_rhs - (ran_lhs @ self.c)

        # convert LHS from np arrays to lists and RHS from np arrays to floats
        dom_lhs = d_lhs_arr
        dom_rhs = d_rhs_arr.flatten()

        # 7) Apply constraint "domain_x<0" to one, "domain_x>=0" to other.
        star_gt_hpoly, star_lt_hpoly = self.hpoly.split_on_halfspace_constraint(dom_lhs, dom_rhs)

        star_gt = StarSet(
            n_dims=self.n_dims,
            c=self.c.copy(),
            g_mat=self.g_mat.copy(),
            hpoly=star_gt_hpoly,
            original_c=self.original_c.copy(),
            original_g_mat=self.original_g_mat.copy(),
            discrete=copy.deepcopy(self.properties),
            transformations=copy.deepcopy(self.transformations),
        )
        star_lt = StarSet(
            n_dims=self.n_dims,
            c=self.c.copy(),
            g_mat=self.g_mat.copy(),
            hpoly=star_lt_hpoly,
            original_c=self.original_c.copy(),
            original_g_mat=self.original_g_mat.copy(),
            discrete=copy.deepcopy(self.properties),
            transformations=copy.deepcopy(self.transformations),
        )
        return star_gt, star_lt

    def get_bounding_box(self):
        extremes = []
        for d in range(self.n_dims):
            # low and high directions in range space
            ran_lo_dir = np.zeros((self.n_dims, 1), dtype=np.float64)
            ran_lo_dir[d, 0] = -1.0
            ran_lo_pt = self.maximize(ran_lo_dir)

            ran_hi_dir = np.zeros((self.n_dims, 1), dtype=np.float64)
            ran_hi_dir[d, 0] = 1.0
            ran_hi_pt = self.maximize(ran_hi_dir)

            extremes.append([float(ran_lo_pt[d]), float(ran_hi_pt[d])])

        rv = bounding_box.BoundingBox(self.n_dims, extremes)
        return rv

    def intersect(self, other):
        assert self.n_dims == other.n_dims

        if isinstance(other, h_polytope.HPoly):
            ran_hpoly_LHS = other.LHS.copy()
            ran_hpoly_rhs = other.rhs.copy()

            # The intersection can inherit its history of transformations
            # from only one of the sets; ensure that only one set has a history
            # of transformations, as there is no sensible way to combine the
            # histories.
            #TODO: assert (not self.transformations) or (not other.transformations)


        elif isinstance(other, bounding_box.BoundingBox):
            ran_hpoly_LHS, ran_hpoly_rhs = other.get_constraints()

        else:
            raise ValueError(f"Intersection of StarSet with {type(other)} not implemented")

        dom_hpoly_LHS = ran_hpoly_LHS @ self.g_mat
        dom_hpoly_rhs = (ran_hpoly_rhs - (ran_hpoly_LHS @ self.c).flatten()).flatten()

        rv = self.copy()
        for dL, dr in zip(dom_hpoly_LHS, dom_hpoly_rhs):
            rv.hpoly.add_le_constraint(dL, dr)
        return rv

    def set_difference(self, other):
        """
        Return the list of parts remaining in this StarSet after removing all
        parts that intersect other.
        If this StarSet does not intersect other, simply return a list
        containing this StarSet.
        """
        assert self.n_dims == other.n_dims

        if isinstance(other, h_polytope.HPoly):
            ran_hpoly_LHS = other.LHS.copy()
            ran_hpoly_rhs = other.rhs.copy()

        elif isinstance(other, bounding_box.BoundingBox):
            ran_hpoly_LHS, ran_hpoly_rhs = other.get_constraints()

        else:
            raise ValueError(f"Set difference of StarSet with {type(other)} not implemented")

        dom_hpoly_LHS = ran_hpoly_LHS @ self.g_mat
        dom_hpoly_rhs = (ran_hpoly_rhs - (ran_hpoly_LHS @ self.c).flatten()).flatten()
        rv_hpolys = self.hpoly._set_difference_with_constrs(dom_hpoly_LHS, dom_hpoly_rhs)

        rv_stars = []
        for hp in rv_hpolys:
            rv_stars.append(self.copy(hpoly=hp))

        return rv_stars

    def check_for_intersection(self, other):
        assert self.n_dims == other.n_dims

        if isinstance(other, h_polytope.HPoly):
            ran_hpoly_LHS = other.LHS.copy()
            ran_hpoly_rhs = other.rhs.copy()

        elif isinstance(other, bounding_box.BoundingBox):
            ran_hpoly_LHS, ran_hpoly_rhs = other.get_constraints()

        else:
            raise ValueError(f"Check for intersection of StarSet with {type(other)} not implemented")

        dom_hpoly_LHS = ran_hpoly_LHS @ self.g_mat
        dom_hpoly_rhs = (ran_hpoly_rhs - (ran_hpoly_LHS @ self.c).flatten()).flatten()

        rv = self.hpoly._check_for_feasibility_with_constrs(dom_hpoly_LHS, dom_hpoly_rhs)
        return rv

    def contains(self, pt):
        assert isinstance(pt, np.ndarray)

        # Use linear programming to check if point is inside star
        normals = self.hpoly.LHS @ np.linalg.inv(self.g_mat)
        offsets = self.hpoly.rhs.copy()

        search_pt = cvxpy.Variable(self.n_dims)

        constraints = []
        for i, normal in enumerate(normals):
            constraints.append(
                # search_pt is inside star set
                (normal.T @ search_pt) <= offsets[i]
            )
        for d in range(self.n_dims):
            tol = 1e-6
            constraints.append(
                search_pt[d] - pt[d] <= tol
            )

        objective = cvxpy.Maximize(None)
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(solver=LP_SOLVER)

        if search_pt.value is None:
            return False
        else:
            return True

    def get_chebyshev_center(self):
        """
        NOTE: This method finds the largest inscribed hypercube of the underlying H-Polytope in the domain,
        then transforms the center of that hypercube to obtain a point in the range. That point in the range
        is returned as the "Chebyshev center" of the Star Set. But we are not actually calculating the inscribed
        hypercube of the Star Set itself, so we don't know the radius of the hypercube.

        TODO: Write a proof (or find a counterexample) of the following assertion:
            By performing an affine transformation on the inscribed hypercube of the HPoly in the domain,
            the center C of the hypercube is transformed to a point C' in the range.
            We could use linear programming to find a point C*, the center of the largest inscribed hypercube
            in the range of the StarSet.
            But there is no need to do so, since C' is always equal to C*, and we have already written a method
            HPoly.get_chebyshev_center() which finds C'.

            The key statement to be proven is: "since C' is always equal to C*"
        """
        _, center = self.hpoly.get_chebyshev_center()
        rv_center = np.dot(self.g_mat, center) + self.c
        return None, rv_center


    def TODO_OLD_CVXPY_get_chebyshev_center(self):
        normals = self.hpoly.LHS @ np.linalg.inv(self.g_mat)
        offsets = self.hpoly.rhs.copy()

        radius = cvxpy.Variable(1)
        center = cvxpy.Variable(self.n_dims)

        constraints = []
        for i, normal in enumerate(normals):
            constraints.append(
                (normal.T @ center) + (np.linalg.norm(normal, ord=2) * radius) <= offsets[i]
            )

        objective = cvxpy.Maximize(radius)
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(solver=LP_SOLVER)

        if radius.value is None or center.value is None:
            raise RuntimeError("Optimization yielded 'None'")
        else:
            center_val = np.asarray(center.value)
            radius_val = radius.value[0]

        center_val = np.reshape(center_val, (self.n_dims, 1))
        center_val += self.c

        return radius_val, center_val

    def undo_all_transformations(self):
        rv = self.copy()
        for (A, b) in reversed(self.transformations):
            try:
                A_inv = np.linalg.inv(A)
            except numpy.linalg.LinAlgError:
                raise ValueError(f"Transformation matrix {A} is not invertible")

            # Perform the opposite of the original affine transformation.
            rv.c -= b
            rv.c = A_inv @ rv.c
            rv.g_mat = A_inv @ rv.g_mat
            rv.transformations = []

        return rv

    @staticmethod
    def from_HPoly(hpoly):
        rv = StarSet(
            n_dims=hpoly.n_dims,
            c=np.zeros((hpoly.n_dims, 1)),
            g_mat=np.eye(hpoly.n_dims),
            hpoly=hpoly,
            discrete=copy.deepcopy(hpoly.properties),
            transformations=copy.deepcopy(hpoly.transformations)
        )
        return rv

    @staticmethod
    def from_BoundingBox(bbox):
        rv_hpoly = h_polytope.HPoly.from_BoundingBox(bbox)
        rv_star = StarSet.from_HPoly(rv_hpoly)
        return rv_star

    @staticmethod
    def from_nnenum_LpStar(nnenum_star):
        """
        Converts an nnenum.lp_star.LpStar into a StarSet
        """
        # This method relies on nnenum libraries that are not used elsewhere.
        import swiglpk as glpk
        import nnenum
        from nnenum.lpinstance import SwigArray

        assert isinstance(nnenum_star, nnenum.lp_star.LpStar), f"Input must be an nnenum.lp_star.LpStar object, instead got {type(nnenum_star)}"

        c = nnenum_star.bias.reshape(-1, 1)
        range_ndims = c.shape[0] # range n_dims

        g_mat = nnenum_star.a_mat
        domain_ndims = g_mat.shape[1] # domain n_dims

        assert g_mat.shape[0] == range_ndims

        def get_bounded_var_intervals(nnenum_star):
            """
            For bounded variables, return the lower and upper bounds defined by the
            .vnnlib input specification. Format of rv is a list of lists, amenable to
            BoundingBox construction:

                [[i0_lb, i0_ub],
                ...
                [iN_lb, iN_ub]]

            Implementation is based on nnenum.lpinstance._var_bounds_str
            """
            intervals = []
            lp = nnenum_star.lpi.lp

            # NOTE: input variables are 1-indexed in the GLPK list (e.g. "i0" at index 1)
            for idx in range(1, domain_ndims+1):
                lb = glpk.glp_get_col_lb(lp, idx)
                ub = glpk.glp_get_col_ub(lp, idx)
                intervals.append([lb, ub])

            return intervals

        ivals = get_bounded_var_intervals(nnenum_star)

        def get_linear_constraints(nnenum_star):
            """
            Returns LHS, rhs representing the linear constraints on the input variables.

            This only includes linear constraints added due to propagation through the
            NN. It does NOT include the input bounds from the .vnnlib specification.
            For those bounds, see get_bounded_var_intervals above.

            Implementation is based on nnenum.lpinstance._constraints_str
            """
            # Copied from nnenum.lpinstance._constraints_str()
            lp = nnenum_star.lpi.lp
            rows = nnenum_star.lpi.get_num_rows()
            cols = nnenum_star.lpi.get_num_cols()

            stat_labels = ["?(0)?", "BS", "NL", "NU", "NF", "NS"]
            inds = SwigArray.get_int_array(cols + 1)
            vals = SwigArray.get_double_array(cols + 1)

            LHS_list = []
            rhs_list = []

            for row in range(1, rows + 1):
                LHS = []
                rhs = None

                stat = glpk.glp_get_row_stat(lp, row)
                assert 0 <= stat <= len(stat_labels)

                num_inds = glpk.glp_get_mat_row(lp, row, inds, vals)
                for col in range(1, cols + 1):
                    val = 0
                    for index in range(1, num_inds+1):
                        if inds[index] == col:
                            val = vals[index]
                            break
                    LHS.append(val)

                row_type = glpk.glp_get_row_type(lp, row)
                assert row_type == glpk.GLP_UP

                rhs = glpk.glp_get_row_ub(lp, row)

                LHS_list.append(LHS)
                rhs_list.append(rhs)

            return (
                np.array(LHS_list),
                np.array(rhs_list),
            )

        LHS, rhs = get_linear_constraints(nnenum_star)

        # Start with the rectangular input bounds
        bbox = bounding_box.BoundingBox(n_dims=domain_ndims, intervals=ivals)

        # Add the lpinstance linear constraints
        hpoly = h_polytope.HPoly.from_BoundingBox(bbox)
        for L, r in zip(LHS, rhs):
            hpoly.add_le_constraint(L, r)

        # Construct the star
        iqv_star = StarSet(
            n_dims=range_ndims,
            c=c,
            g_mat=g_mat,
            hpoly=hpoly,
        )
        return iqv_star


    def remove_redundant_constraints(self):
       self.hpoly.remove_redundant_constraints()


    def minkowski_sum(self, other):
        """
        References:
        - Bak, Stanley, and Parasara Sridhar Duggirala. "Simulation-equivalent
        reachability of large linear systems with inputs." International
        Conference on Computer Aided Verification. Cham: Springer International
        Publishing, 2017. https://stanleybak.com/papers/bak2017cav.pdf

        - Raghuraman, Vignesh, and Justin P. Koeln. "Set operations and order
        reductions for constrained zonotopes." Automatica 139 (2022): 110204.

        Explanation:
        A side-effect of the Minkowski sum is that the dimension of the H-poly
        will increase, as will the size (number of columns, only) of the
        generator matrix. (Number of rows == number of star set dims.)
        As the generator matrix gains more columns, it is capable of converting
        from the lower dimension star set (range), to the higher dimension
        H-poly (domain), and vice versa.
        In the example of a maximization (e.g. for plotting), the user gives
        a direction in the range. The generator matrix converts this to a
        direction in the domain, which is a higher dimension after performing
        the Minkowski sum. We perform the LP in higher (domain) dimension to
        obtain a point in the domain. The generator matrix once again converts
        this, to a point in the range (lower dimension).
        """
        assert isinstance(other, StarSet)
        assert self.n_dims == other.n_dims

        rv_c = self.c + other.c

        # rv_G = [G1  G2]
        rv_g_mat = np.hstack((self.g_mat, other.g_mat))

        # rv_LHS = [[LHS1    0 ... 0]
        #           [0 ... 0  LHS2  ]]
        rv_LHS_top = np.hstack((self.hpoly.LHS, np.zeros((self.hpoly.LHS.shape[0], other.hpoly.LHS.shape[1]))))
        rv_LHS_bot = np.hstack((np.zeros((other.hpoly.LHS.shape[0], self.hpoly.LHS.shape[1])), other.hpoly.LHS))
        rv_LHS = np.vstack((rv_LHS_top, rv_LHS_bot))

        # rv_rhs = [[rhs1]
        #           [rhs2]]
        rv_rhs = np.hstack((self.hpoly.rhs, other.hpoly.rhs))

        # n_dims of the underlying HPoly increases. Any LPs will be in the
        # higher dimension, due to the extra "0s" concatenated in the LHS. The
        rv_hpoly = h_polytope.HPoly(
            n_dims=self.hpoly.LHS.shape[1] + other.hpoly.LHS.shape[1],
            LHS=rv_LHS,
            rhs=rv_rhs,
        )

        # n_dims of the resulting StarSet will still match the input StarSets.
        rv = StarSet(
            n_dims=self.n_dims,
            g_mat=rv_g_mat,
            c=rv_c,
            hpoly=rv_hpoly,
            discrete=copy.deepcopy(self.properties),
            transformations=copy.deepcopy(self.transformations),
        )

        return rv

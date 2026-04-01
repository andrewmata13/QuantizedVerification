import numpy as np
import matplotlib.pyplot as plt
import copy
import logging
log = logging.getLogger(__name__)
import gurobipy as gp
import cvxpy

from iq_verify.set_repr import set_repr, bounding_box, v_polytope, star_set, point_repr, star_zono_prefilter
from iq_verify.linprog import linprog

N_PLOTTING_DIRS = 100
DIMS_TO_PLOT = (0, 1)

class InfeasibleException(Exception):
    pass

class HPoly(set_repr.SetRepr):
    def __init__(self, n_dims, LHS, rhs, transformations=None, discrete=None, LP=None):
        super().__init__(n_dims, discrete)
        self.LHS = LHS
        self.rhs = rhs
        assert isinstance(self.LHS, np.ndarray) and self.LHS.shape[1] == self.n_dims, f"Expected LHS to be a matrix with {self.n_dims} columns, got {self.LHS.shape}"
        assert isinstance(self.rhs, np.ndarray) and self.rhs.shape == (self.LHS.shape[0],), f"Expected rhs to contain one value per lhs row (LHS is {self.LHS.shape}), got {self.rhs.shape}"

        # Track the affine transformations (A, b) that have been performed
        # on this HPoly, in case we need to "undo" them later. For example,
        # when forward reachability reveals an intersection with the unsafe set,
        # we may wish to trace the intersection back to its original location.
        self.transformations = transformations
        if self.transformations is None:
            self.transformations = []

        if LP is None:
            # Convert constraint matrix + vector into LP constraints
            ####self.LP = linprog.LinProgGurobi(n_dims=self.n_dims)
            self.LP = linprog.LinProgGLPK(n_dims=self.n_dims)
            self.LP.init_LHS_rhs_constraints(self.LHS, self.rhs)
        else:
            self.LP = LP

    def plot(self, color, fill=False, alpha=1.0):
        """Plot the set"""
        err_msg = f"Cannot plot infeasible HPoly"
        if self.check_is_feasible() == False:
            log.warn(err_msg)
            return

        try:
            vp = self.convert_to_vpoly()
        except (InfeasibleException, gp.GurobiError):
            log.warn(err_msg)
            return

        vp = vp.project_to_2D(i=DIMS_TO_PLOT[0], j=DIMS_TO_PLOT[1])
        vp.plot(color, fill, alpha)

    def maximize(self, dir_vec):
        """Return the set's extreme point in the given direction"""
        self.LP.update_model()

        # use LP to optimize the constraints
        obj_vec = dir_vec.flatten()
        self.LP.maximize(obj_vec)

        if self.LP.model_infeasible():
            raise InfeasibleException()
        elif self.LP.optimization_failed():
            log.warn(f"Optimization failed")
            raise InfeasibleException("Optimization failed")

        rv = self.LP.get_solution()
        assert rv.shape == (self.n_dims, 1)
        return rv

    def get_verts(self):
        """Return the vertices of the set's convex hull"""
        vp = self.convert_to_vpoly()
        return vp.get_verts()

    def convert_to_vpoly(self):
        sorted_verts = []
        for _ in range(N_PLOTTING_DIRS):
            dir_vec = np.array([
                [np.random.default_rng().uniform(low=-1.0, high=1.0)]
                for d in range(self.n_dims)
            ])
            v = self.maximize(dir_vec)
            sorted_verts.append(v)
        rv = v_polytope.VPoly(n_dims=self.n_dims, verts=sorted_verts)
        return rv

    def affine_transform(self, A, b, in_place=False):
        """Return this set after the affine transformation"""
        # When the linear transformation matrix A is left-invertible, we can
        # perform an affine transformation on an H-Polytope.
        #
        #     HPoly: Cx <= d
        #     Affine Transformation: y = Ax + b
        #     ...solve for x...
        #     Transformed HPoly: C @ A_inv @ y <= d + (C @ A_inv @ b)
        #
        # See e.g. Equation 4 here: https://arxiv.org/pdf/1903.05214
        #
        # In the case of reachability analysis, this is a safe assumption, since
        # the matrix exponential e^(At), or its backward counterpart e^(-At),
        # is always invertible. See e.g. https://en.wikipedia.org/wiki/Matrix_exponential
        try:
            A_inv = np.linalg.inv(A)
        except numpy.linalg.LinAlgError:
            raise ValueError(f"Transformation matrix {A} is not invertible")

        # TODO: hard-coded, for now. Would need to transform constraints inside
        # LP model as well; simpler to reconstruct from scratch.
        in_place = False

        rv = HPoly(
            n_dims=self.n_dims,
            LHS=self.LHS @ A_inv,
            rhs=self.rhs + (self.LHS @ A_inv @ b).flatten(),
            transformations=self.transformations + [(A, b)],
            discrete=copy.deepcopy(self.properties),
        )

        return rv

    def add_le_constraint(self, lhs, rhs):
        """Update the constraint matrix + vector in both the HPoly and the LP solver"""
        self.LHS = np.vstack((self.LHS, lhs))
        self.rhs = np.hstack((self.rhs, rhs))
        self.LP.add_le_constraint(lhs, rhs)

    def split_on_halfspace_constraint(self, lhs, rhs):
        """Return the (greater_than, less_than) portions of this stateset with respect to the constraint"""
        left = self.copy()
        left.add_le_constraint(lhs, rhs)

        right = self.copy()
        right.add_le_constraint(-lhs, -rhs)

        return (right, left)

    def get_bounding_box(self):
        """Returning a bounding box that contains this stateset"""
        intervals = []
        for i in range(self.n_dims):
            lo_dir = np.zeros((self.n_dims, 1))
            lo_dir[i, 0] = -1.0
            hi_dir = np.zeros((self.n_dims, 1))
            hi_dir[i, 0] = 1.0
            lo = self.maximize(lo_dir)[i][0]
            hi = self.maximize(hi_dir)[i][0]
            intervals.append([lo, hi])

        rv = bounding_box.BoundingBox(n_dims=self.n_dims, intervals=intervals)
        return rv

    @staticmethod
    def from_BoundingBox(bbox):
        LHS, rhs = bbox.get_constraints()

        rv = HPoly(
            n_dims=bbox.n_dims,
            LHS=LHS,
            rhs=rhs,
            discrete=copy.deepcopy(bbox.properties)
        )

        return rv

    def __str__(self):
        """Return a string representation of this set"""
        rv = f"HPoly:\nn_dims:{self.n_dims}\nLHS:"
        for l in self.LHS:
            rv += f"\n\t{l}"
        rv += f"\nrhs: {self.rhs}"
        return rv

    def copy(self):
        """Return a copy of this set"""
        rv = HPoly(
            n_dims=self.n_dims,
            LHS=copy.deepcopy(self.LHS),
            rhs=copy.deepcopy(self.rhs),
            transformations=copy.deepcopy(self.transformations),
            discrete=copy.deepcopy(self.properties),
            LP=self.LP.copy()
        )
        return rv

    def check_is_feasible(self):
        # Test whether we can optimize in any direction
        random_dir = np.ones((self.n_dims, )) * np.random.rand(self.n_dims)
        try:
            self.maximize(random_dir)
        except InfeasibleException:
            return False

        return True

    def contains(self, pt):
        assert isinstance(pt, np.ndarray)
        pt = pt.flatten()
        lhs = self.LHS @ pt
        rv = (lhs <= self.rhs).all()
        return rv

    def _check_for_feasibility_with_constrs(self, LHS, rhs):
        """
        Checks whether adding the constraints (LHS, rhs) would hypothetically
        yield a feasible HPoly, but without actually calculating the resulting HPoly.
        """
        # To avoid constructing a new HPoly, we will modify this HPoly.
        # Create a backup of this HPoly.
        self_LHS_copy = self.LHS.copy()
        self_rhs_copy = self.rhs.copy()

        # Add the hypothetical constraints and see if the resulting HPoly is feasible.
        for (L, r) in zip(LHS, rhs):
            self.add_le_constraint(L, r)
        rv = self.check_is_feasible()

        # Restore this HPoly to the backup.
        #
        # Note that add_le_constraint modifies the LHS and rhs arrays,
        # as well as the underlying LP object. Reverting the arrays does
        # not automatically revert the LP object; we must manually remove
        # the added constraints.
        self.LHS = self_LHS_copy
        self.rhs = self_rhs_copy
        for _ in zip(LHS, rhs):
            self.LP.pop_le_constraint()

        return rv

    def check_for_intersection(self, other):
        assert self.n_dims == other.n_dims

        if isinstance(other, HPoly):
            rv = self._check_for_feasibility_with_constrs(other.LHS, other.rhs)
            return rv

        elif isinstance(other, bounding_box.BoundingBox):
            LHS, rhs = other.get_constraints()
            rv = self._check_for_feasibility_with_constrs(LHS, rhs)
            return rv

        elif isinstance(other, point_repr.Point):
            return self.intersect(other)

        elif isinstance(other, star_set.StarSet):
            return other.check_for_intersection(self)

        elif isinstance(other, star_zono_prefilter.StarZonoPrefilter):
            return other.check_for_intersection(self)

        else:
            raise ValueError(f"Intersection checking for HPoly with {type(other)} not implemented")

    def intersect(self, other):
        if isinstance(other, HPoly) or isinstance(other, bounding_box.BoundingBox):
            if isinstance(other, HPoly):
                LHS, rhs = (other.LHS, other.rhs)
            elif isinstance(other, bounding_box.BoundingBox):
                LHS, rhs = other.get_constraints()

            rv = self.copy()
            for L, r in zip(LHS, rhs):
                rv.add_le_constraint(L, r)
            rv.LP.update_model()

            # The intersection can inherit its history of transformations
            # from only one of the sets; ensure that only one set has a history
            # of transformations, as there is no sensible way to combine the
            # histories.
            assert (not self.transformations) or (not other.transformations)
            if self.transformations:
                rv.transformations = self.transformations
            else:
                rv.transformations = other.transformations

        elif isinstance(other, point_repr.Point):
            is_contained = self.contains(other.coords)
            if is_contained:
                rv = other
            else:
                rv = HPoly(
                    n_dims=self.n_dims,
                    LHS=np.zeros((1, self.n_dims)),
                    rhs=np.zeros((1, )),
                )

        elif isinstance(other, star_set.StarSet):
            rv = other.intersect(self)

        elif isinstance(other, star_zono_prefilter.StarZonoPrefilter):
            rv = other.intersect(self)

        else:
            raise ValueError(f"Intersection of HPoly with {type(other)} not implemented")

        return rv

    def get_chebyshev_center(self):
        '''
        # TODO: This method is hard-coded to use gurobipy
        '''
        normals = self.LHS.copy()
        offsets = self.rhs.copy()

        model = gp.Model()
        model.update()
        model.setParam("OutputFlag", 0)
        model.update()

        center = []
        for d in range(self.n_dims):
            ci = model.addVar(lb=-np.inf, ub=np.inf, vtype=gp.GRB.CONTINUOUS, name=f"c_{d}")
            center.append(ci)

        radius = model.addVar(lb=-np.inf, ub=np.inf, vtype=gp.GRB.CONTINUOUS, name=f"radius")

        model.update()

        for normal, offset in zip(normals, offsets):
            model.addLConstr(
                lhs=(normal.T @ center) + (np.linalg.norm(normal, ord=2) * radius),
                sense=gp.GRB.LESS_EQUAL,
                rhs=offset,
            )
            model.update()

        # use LP to optimize the constraints
        model.setObjective(radius, gp.GRB.MAXIMIZE)
        model.update()
        model.optimize()
        model.update()

        if model.status in [gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD]:
            raise InfeasibleException()
        elif model.status != gp.GRB.OPTIMAL:
            log.warn(f"GUROBI optimization failed with status: {model.status}")
            raise gp.GurobiError(-1, "GUROBI optimization failed")

        rv_radius = radius.x
        rv_center = np.array([c.x for c in center]).reshape((self.n_dims, 1))

        return rv_radius, rv_center

    def TODO_OLD_get_chebyshev_center(self):
        normals = self.LHS.copy()
        offsets = self.rhs.copy()

        # Chebyshev center

        radius = cvxpy.Variable(1)
        center = cvxpy.Variable(self.n_dims)

        constraints = []
        for offset, normal in zip(offsets, normals):
            constraints.append(
                (normal.T @ center) + (np.linalg.norm(normal, ord=2) * radius) <= offset
            )

        objective = cvxpy.Maximize(radius)
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(solver=cvxpy.GUROBI)

        if radius.value is None or center.value is None:
            raise RuntimeError("Optimization yielded 'None'")

        center_val = np.asarray(center.value)
        center_val = np.reshape(center_val, (self.n_dims, 1))
        radius_val = radius.value[0]

        return radius_val, center_val

    def set_difference(self, other):
        if isinstance(other, HPoly):
            rv = self._set_difference_with_constrs(other.LHS, other.rhs)
            return rv

        elif isinstance(other, bounding_box.BoundingBox):
            LHS, rhs = other.get_constraints()
            rv = self._set_difference_with_constrs(LHS, rhs)
            return rv

        else:
            raise ValueError(f"Set difference of HPoly with {type(other)} not implemented")

    def _set_difference_with_constrs(self, LHS, rhs):
        if not self._check_for_feasibility_with_constrs(LHS, rhs):
            return [self]

        sets = []
        inside = self.copy()
        for L, r in zip(LHS, rhs):
            if not self._check_for_feasibility_with_constrs([-L], [-r]):
                continue

            outside = inside.copy()
            outside.add_le_constraint(-L, -r)
            if outside.check_is_feasible():
                sets.append(outside)

            inside.add_le_constraint(L, r)
        return sets

    def undo_all_transformations(self):
        """Undo all affine transformations performed on this HPoly"""
        rv = self.copy()
        for (A, b) in reversed(self.transformations):
            try:
                A_inv = np.linalg.inv(A)
            except numpy.linalg.LinAlgError:
                raise ValueError(f"Transformation matrix {A} is not invertible")

            # Perform the opposite of the original affine transformation.
            reverted_LHS = rv.LHS @ A
            reverted_rhs = rv.rhs - (reverted_LHS @ A_inv @ b).flatten()

            rv = HPoly(
                n_dims=self.n_dims,
                LHS=reverted_LHS,
                rhs=reverted_rhs,
                transformations=None,
            )

        return rv

    def remove_redundant_constraints(self):
        keep_going = True
        while keep_going:
            keep_going = False

            for i, (C, d) in enumerate(zip(self.LHS, self.rhs)):
                # Maximize in the direction of the candidate constraint
                vert = self.maximize(C)

                # Remove the constraint and test to see if the same vertex is still an extreme
                _LHS = self.LHS.copy()
                _rhs = self.rhs.copy()
                _LHS = np.delete(_LHS, i, axis=0)
                _rhs = np.delete(_rhs, i, axis=0)
                _hpoly = HPoly(
                    n_dims=self.n_dims,
                    LHS=_LHS,
                    rhs=_rhs,
                )
                _vert = _hpoly.maximize(C)

                if np.all(np.isclose(vert, _vert)):
                    # Constraint is redundant; remove it from the HPoly
                    self.LHS = _LHS
                    self.rhs = _rhs

                    # Restart, so that we are not removing from the array while iterating over it
                    keep_going = True
                    break

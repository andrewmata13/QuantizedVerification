import copy
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import itertools
import logging
log = logging.getLogger(__name__)

# TODO: Remove cvxpy, use gurobipy directly
import cvxpy
#LP_SOLVER = cvxpy.GLPK_MI  # NOTE: GLPK_MI failed miserably at finding the Chebyshev center of a StarSet
LP_SOLVER = cvxpy.GUROBI
import pyeda

# TODO: Remove all scipy LPs
import scipy

import set_repr
import bounding_box
from bdd_set import BDDset
import kamenev

N_PLOTTING_DIRS = 10
N_FEASIBILITY_DIRS = 1


class InfeasibleException(Exception):
    # TODO: Copied from H-Polytope
    pass


class LCDD(set_repr.SetRepr):
    def __init__(self, n_dims, *, bddset=None, ex=None, C=None, d=None, var_list=None, transformations=None):
        super().__init__(n_dims)

        # Must provide either a BDDset, or all the information required to construct one.
        # Must explicitly provide named parameters beyond n_dims.
        if bddset is None:
            # var_list is optional
            assert (ex is not None) and (C is not None) and (d is not None)
            self.bddset = BDDset.BDDset(ex, C, d, var_list)
        else:
            self.bddset = bddset

        self.n_dims = n_dims
        assert self.n_dims == self.bddset.C.shape[1]

        # Track the affine transformations (A, b) that have been performed
        # on this LCDD, in case we need to "undo" them later. For example,
        # when forward reachability reveals an intersection with the unsafe set,
        # we may wish to trace the intersection back to its original location.
        self.transformations = transformations
        if self.transformations is None:
            self.transformations = []


    def plot(self, color, fill=None, alpha=0.3):
        self.bddset.plot(color=color, show=False, alpha=alpha)


    def affine_transform(self, A, b):
        assert b.shape == (A.shape[0], 1), f"b has shape {b.shape} instead of {(A.shape[0], 1)}"
        new_bddset = self.bddset.affineMap(A=A, b=b)
        rv = LCDD(
            n_dims=self.n_dims,
            bddset=new_bddset,
            transformations=self.transformations + [(A, b)],
        )
        return rv


    def split_on_halfspace_constraint(self, lhs, rhs):
        rhs = rhs.reshape((lhs.shape[0], 1))
        ex = pyeda.inter.expr("z")

        gt_halfspace = BDDset.BDDset(ex=ex, C=-lhs, d=-rhs, var_list=None)
        gt_bdd = self.bddset.__and__(gt_halfspace)

        lt_halfspace = BDDset.BDDset(ex=ex, C=lhs, d=rhs, var_list=None)
        lt_bdd = self.bddset.__and__(lt_halfspace)

        gt_rv = LCDD(self.n_dims, bddset=gt_bdd, transformations=self.transformations)
        lt_rv = LCDD(self.n_dims, bddset=lt_bdd, transformations=self.transformations)

        return (gt_rv, lt_rv)


    def path_to_constraints(self, path_dict):
        # Keep only the subset of constraints corresponding to variables assigned True.
        C_list = []
        d_list = []

        for var, val in path_dict.items():
            i = self.bddset.ordered_inputs.index(var)

            if val == 1:
                C_list.append(self.bddset.C[i])
                d_list.append(self.bddset.d[i])
            else:
                C_list.append(-1 * self.bddset.C[i])
                d_list.append(-1 * self.bddset.d[i])

        C_mat = np.array(C_list)
        d_vec = np.array(d_list)
        return (C_mat, d_vec)


    def get_verts(self):
        all_verts = []

        # All paths from BDD entrypoint to the terminal "1" node.
        all_satisfying_paths = self.bddset.bdd.satisfy_all()

        for path_dict in all_satisfying_paths:
            C_mat, d_vec = self.path_to_constraints(path_dict)

            def dir_to_pt(dir_vec):
                pt = self.maximize(dir_vec, C_mat, d_vec)
                return pt.flatten() # kamenev requires 1D array

            try:
                verts = kamenev.get_verts(self.n_dims, dir_to_pt, epsilon=1e-6)
            except InfeasibleException:
                continue

            if isinstance(verts, list):
                all_verts += verts
            elif isinstance(verts, np.ndarray):
                all_verts += verts.tolist()

        unique_verts = list(set(map(tuple, all_verts)))
        return unique_verts


    def get_bounding_box(self):
        intervals = [
            [None, None] for _ in range(self.n_dims)
        ]
        for v in self.get_verts():
            for d in range(self.n_dims):
                min_so_far = intervals[d][0]
                if min_so_far is None or v[d] < min_so_far:
                    intervals[d][0] = v[d]

                max_so_far = intervals[d][1]
                if max_so_far is None or v[d] > max_so_far:
                    intervals[d][1] = v[d]

        rv = bounding_box.BoundingBox(n_dims=self.n_dims, intervals=intervals)
        return rv


    def check_is_feasible(self):
        for satisfying_path in self.bddset.bdd.satisfy_all():
            C_mat, d_vec = self.path_to_constraints(satisfying_path)
            if self.heuristically_check_constraints_are_feasible(C_mat, d_vec):
                return True
        return False


    def __str__(self):
        return "LCDD " + str(self.bddset)


    def intersect(self, other):
        rv_bddset = self.bddset.__and__(other.bddset)
        rv = LCDD(n_dims=self.n_dims, bddset=rv_bddset)

        # The intersection can inherit its history of transformations
        # from only one of the sets; ensure that only one set has a history
        # of transformations, as there is no sensible way to combine the
        # histories.
        assert (not self.transformations) or (not other.transformations)
        if self.transformations:
            rv.transformations = self.transformations
        else:
            rv.transformations = other.transformations

        return rv


    def set_difference(self, other):
        rv_bddset = self.bddset.setdiff(other.bddset)
        rv = LCDD(n_dims=self.n_dims, bddset=rv_bddset)

        # The set difference can inherit its history of transformations
        # from only one of the sets; ensure that only one set has a history
        # of transformations, as there is no sensible way to combine the
        # histories.
        assert (not self.transformations) or (not other.transformations)
        if self.transformations:
            rv.transformations = self.transformations
        else:
            rv.transformations = other.transformations

        return [rv]


    def undo_all_transformations(self):
        """Undo all affine transformations performed on this HPoly"""
        rv_bddset = BDDset.BDDset(
            ex=self.bddset.bdd,
            C=self.bddset.C,
            d=self.bddset.d,
            var_list=self.bddset.var_list,
        )

        for (A, b) in reversed(self.transformations):
            try:
                A_inv = np.linalg.inv(A)
            except numpy.linalg.LinAlgError:
                raise ValueError(f"Transformation matrix {A} is not invertible")

            rv_bddset += -b
            rv_bddset *= A_inv

        rv = LCDD(
            n_dims=self.n_dims,
            bddset=rv_bddset,
        )
        return rv


    def get_chebyshev_center(self):
        for satisfying_path in self.bddset.bdd.satisfy_all():
            C_mat, d_vec = self.path_to_constraints(satisfying_path)
            if self.heuristically_check_constraints_are_feasible(C_mat, d_vec):
                r, c = self._chebyshev_center_LP(C_mat, d_vec)
                return r, c

        # Failed to find any feasible subcomponent
        return None, None


    def contains(self, pt):
        rv = self.bddset.contains(pt.flatten())
        return rv


    def maximize(self, dir_vec, C_mat, d_vec):
        # TODO: Copied almost verbatim from H-Polytope, except HPoly takes C,d from fields.
        # Should make this a 3-argument method of the SetRepr class.
        """Return the set's extreme point in the given direction"""
        model = gp.Model()
        model.update()
        model.setParam("OutputFlag", 0)
        model.update()
        for d in range(self.n_dims):
            model.addVar(lb=-np.inf, ub=np.inf, vtype=gp.GRB.CONTINUOUS, name=f"x_{d}")
        model.update()

        for L, r in zip(C_mat, d_vec):
            model.addLConstr(
                lhs=gp.LinExpr(L, model.getVars()),
                sense=gp.GRB.LESS_EQUAL,
                rhs=r,
            )
            model.update()

        # use LP to optimize the constraints
        model.setObjective(dir_vec.flatten() @ model.getVars(), gp.GRB.MAXIMIZE)
        model.update()
        model.optimize()
        model.update()

        if model.status in [gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD]:
            raise InfeasibleException()
        elif model.status != gp.GRB.OPTIMAL:
            log.warn(f"GUROBI optimization failed with status: {model.status}")
            raise gp.GurobiError(-1, "GUROBI optimization failed")

        rv = []
        for d in range(self.n_dims):
            rv.append(model.getVars()[d].x)

        return np.array(rv).reshape((self.n_dims, 1))


    def heuristically_check_constraints_are_feasible(self, C_mat, d_vec):
        for _ in range(N_FEASIBILITY_DIRS):
            rand_dir = np.random.default_rng().uniform(size=(self.n_dims,), low=-1.0, high=1.0)
            try:
                pt = self.maximize(rand_dir, C_mat, d_vec)
            except (AssertionError, InfeasibleException) as err:
                return False
            except gp.GurobiError as err:
                print("TODO: A serious gurobipy error has occurred")
                return False
        return True


    def _chebyshev_center_LP(self, C_mat, d_vec):
        # TODO: Copied almost verbatim from HPolytope, except the assignment to normals, offsets.
        # Should make a top-level function in set_repr.
        normals = C_mat.copy()
        offsets = d_vec.copy()

        # Chebyshev center

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

        center_val = np.asarray(center.value)
        center_val = np.reshape(center_val, (self.n_dims, 1))
        radius_val = radius.value[0]

        return radius_val, center_val

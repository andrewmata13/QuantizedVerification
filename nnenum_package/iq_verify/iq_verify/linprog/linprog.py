import abc
import copy
import gurobipy as gp
import numpy as np
import swiglpk as glpk

class LinProg(abc.ABC):
    @abc.abstractmethod
    def __init__(self, n_dims):
        return

    @abc.abstractmethod
    def update_model(self):
        return

    @abc.abstractmethod
    def add_le_constraint(self, L, r):
        return

    @abc.abstractmethod
    def pop_le_constraint(self):
        """
        Remove the most recently added constraint.
        """
        return

    @abc.abstractmethod
    def init_LHS_rhs_constraints(self, LHS, rhs):
        return

    @abc.abstractmethod
    def maximize(self, obj_vec):
        return

    @abc.abstractmethod
    def model_infeasible(self):
        return

    @abc.abstractmethod
    def optimization_failed(self):
        return

    @abc.abstractmethod
    def get_solution(self):
        return

    @abc.abstractmethod
    def copy(self):
        return


class LinProgGurobi(LinProg):
    def __init__(self, n_dims, model=None):
        self.n_dims = n_dims
        if model is not None:
            self.model = model
            self.xi_vars = model.getVars()
        else:
            self.model = gp.Model()
            self.model.setParam("OutputFlag", 0)
            self.xi_vars = self.init_xi_foreach_dim()

    def copy(self):
        self.model.update()
        rv_model = self.model.copy()
        rv_model.update()
        rv = LinProgGurobi(self.n_dims, model=rv_model)
        return rv

    def update_model(self):
        self.model.update()

    def init_xi_foreach_dim(self):
        xi_vars = []
        for d in range(self.n_dims):
            v = self.model.addVar(lb=-np.inf, ub=np.inf, vtype=gp.GRB.CONTINUOUS, name=f"x_{d}")
            xi_vars.append(v)
        self.update_model()
        return xi_vars

    def add_le_constraint(self, L, r):
        self.model.addLConstr(
            lhs=gp.LinExpr(L, self.xi_vars),
            sense=gp.GRB.LESS_EQUAL,
            rhs=r,
        )

    def pop_le_constraint(self):
        constr_to_pop = self.model.getConstrs()[-1]
        self.model.remove(constr_to_pop)
        self.update_model()

    def init_LHS_rhs_constraints(self, LHS, rhs):
        for L, r in zip(LHS, rhs):
            self.add_le_constraint(L, r)

    def maximize(self, obj_vec):
        self.model.setObjective(obj_vec @ self.xi_vars, gp.GRB.MAXIMIZE)
        self.model.optimize()

    def model_infeasible(self):
        rv = self.model.status in [gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD]
        return rv

    def optimization_failed(self):
        rv = self.model.status != gp.GRB.OPTIMAL
        return rv

    def get_solution(self):
        rv = []
        for d in range(self.n_dims):
            rv.append(self.model.getVars()[d].x)

        rv_vec = np.array(rv).reshape((self.n_dims, 1))
        return rv_vec

class LinProgGLPK(LinProg):
    def __init__(self, n_dims):
        from iq_verify.linprog import lpinstance

        self.n_dims = n_dims
        self.model = lpinstance.LpInstance()
        self.init_xi_foreach_dim()
        self.result = None

    def init_xi_foreach_dim(self):
        xi_vars = []
        for d in range(self.n_dims):
            xi_vars.append(f"x_{d}")
        self.model.add_cols(xi_vars)

    def update_model(self):
        pass

    def add_le_constraint(self, L, r):
        self.model.add_dense_row(L, float(r))

    def pop_le_constraint(self):
        row_idx = glpk.glp_get_num_rows(self.model.lp)
        rows_arr = glpk.intArray(1)
        rows_arr[1] = row_idx
        glpk.glp_del_rows(self.model.lp, 1, rows_arr)
        glpk.glp_std_basis(self.model.lp) # doing this prevents a long stream of warning logs

    def init_LHS_rhs_constraints(self, LHS, rhs):
        for L, r in zip(LHS, rhs):
            self.model.add_dense_row(L, r)

    def maximize(self, obj_vec):
        self.result = self.model.minimize(-1 * obj_vec, fail_on_unsat=False)

    def model_infeasible(self):
        rv = self.result is None
        return rv

    def optimization_failed(self):
        rv = self.result is None
        return rv

    def get_solution(self):
        rv = self.result.reshape(self.n_dims, 1)
        return rv

    def copy(self):
        return copy.deepcopy(self)

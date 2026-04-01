from typing import Tuple

from subprocess import check_call
import tempfile
import os

from copy import deepcopy
from scipy.optimize import linprog

from pyeda.inter import *
from pyeda import *
import numpy as np
from sets.Polytope import Polytope

import kamenev

from z3 import *

from sys import platform
if platform == 'darwin':
    import matplotlib
    matplotlib.use('MacOSX')

import matplotlib.pyplot as plt

class BDDset:
    """class representing sets via binary decision diagrams"""

    def __init__(self, ex, C, d, var_list=None):
        """class constructor

        if var_list is omitted, the variables in ex are assumted to be in alphabetical order
        """

        assert C.shape[0] == d.shape[0], 'Dimensions of C and d are not compatible'

        if isinstance(ex, BinaryDecisionDiagram):
            self.bdd = ex
        else:
            if not isinstance(ex, Expression):
                ex = expr(ex)

            self.bdd = expr2bdd(ex)

        self.ordered_inputs : Tuple[BDDVariable] = tuple()

        if var_list is None:
            # assume variables are in alphabetical order
            self.ordered_inputs = tuple(i for i in sorted(self.bdd.inputs))

            assert len(self.bdd.inputs) == d.shape[0], f"bdd has {len(self.bdd.inputs)} vars, " + \
                f"C has {C.shape[0]} rows"

            self.C = C.copy()
            self.d = d.copy()
        else:
            # var_list was provided
            assert len(var_list) == d.shape[0], "var_list should have same length as C and d"
            assert len(var_list) == 0 or isinstance(var_list[0], str), f"var_list should contain strings: {var_list}"

            # first delete variables in var_list,C,d that are not needed
            bdd_input_names = set(i.name for i in self.bdd.inputs)
            del_indices = []

            for i, var in enumerate(var_list):
                if var not in bdd_input_names:
                    del_indices.append(i)

            var_list = list(np.delete(np.array(var_list), del_indices))
            self.C = np.delete(C, del_indices, axis=0)
            self.d = np.delete(d, del_indices, axis=0)

            # created ordered_inputs based on var_list
            self.ordered_inputs = [None] * self.d.shape[0]

            for bdd_input in self.bdd.inputs:
                i = var_list.index(bdd_input.name)
                self.ordered_inputs[i] = bdd_input

            assert None not in self.ordered_inputs

        self.var_list = self.make_var_list()
        self.uniqid_list = self.make_uniqid_list()

        assert len(self.bdd.inputs) == self.C.shape[0], f'bdd has {len(self.bdd.inputs)} inputs, C has {C.shape[0]} rows'

    def __and__(self, other_set):
        """intersection of two sets"""

        other_set = self.unique_var_names(other_set)

        C = np.concatenate((self.C, other_set.C), axis=0)
        d = np.concatenate((self.d, other_set.d), axis=0)
        bdd = self.bdd & other_set.bdd
        var_list = self.var_list + other_set.var_list

        return BDDset(bdd, C, d, var_list)

    def __or__(self, other_set):
        """union of two sets"""

        other_set = self.unique_var_names(other_set)

        C = np.concatenate((self.C, other_set.C), axis=0)
        d = np.concatenate((self.d, other_set.d), axis=0)
        bdd = self.bdd | other_set.bdd
        var_list = self.var_list + other_set.var_list

        return BDDset(bdd, C, d, var_list)

    def constraint_preserving_union(self, other_set):
        """union of two sets, without unique var names"""

        if len(self.var_list) >= len(other_set.var_list):
            bigger_set = self
            smaller_set = other_set
        else:
            bigger_set = self
            smaller_set = other_set

        for smaller_index, var in enumerate(smaller_set.var_list):
            assert var in bigger_set.var_list, f"var_lists are incomparable: self.var_list was {self.var_list}, " + \
              f"other_set.var_list was: {other_set.var_list}"

            bigger_index = bigger_set.var_list.index(var)

            assert np.allclose(smaller_set.C[smaller_index], bigger_set.C[bigger_index])
            assert np.allclose(smaller_set.d[smaller_index], bigger_set.d[bigger_index])

        bdd = self.bdd | other_set.bdd

        return BDDset(bdd, bigger_set.C, bigger_set.d, bigger_set.var_list)

    def constraint_preserving_intersection(self, other_set):
        """intersection of two sets, without unique var names"""

        if len(self.var_list) >= len(other_set.var_list):
            bigger_set = self
            smaller_set = other_set
        else:
            bigger_set = self
            smaller_set = other_set

        for smaller_index, var in enumerate(smaller_set.var_list):
            assert var in bigger_set.var_list, f"var_lists are incomparable: self.var_list was {self.var_list}, " + \
              f"other_set.var_list was: {other_set.var_list}"

            bigger_index = bigger_set.var_list.index(var)

            print(smaller_set.C)
            print(bigger_set.C)
            assert np.allclose(smaller_set.C[smaller_index], bigger_set.C[bigger_index])
            assert np.allclose(smaller_set.d[smaller_index], bigger_set.d[bigger_index])

        bdd = self.bdd & other_set.bdd

        return BDDset(bdd, bigger_set.C, bigger_set.d, bigger_set.var_list)

    def __mul__(self, A):
        """linear map S * A of set S with matrix A"""

        C = np.dot(self.C, np.linalg.inv(A))

        return BDDset(self.bdd, C, self.d, self.var_list)

    def __add__(self, b):
        """linear map S + b of set S with vector b"""

        d = self.d + np.dot(self.C, b)

        return BDDset(self.bdd, self.C, d, self.var_list)

    def make_var_list(self):
        """get the ordered list of variable names"""

        return [i.name for i in self.ordered_inputs]

    def make_uniqid_list(self):
        """get the ordered list of variable names"""

        return [i.uniqid for i in self.ordered_inputs]

    def as_z3_expression(self, node=None, inverse=False):
        """get the bdd node as an exact z3 expression"""

        if node is None:
            node = self.bdd.node

        if node is boolalg.bdd.BDDNODEONE:
            rv = not inverse
        elif node == boolalg.bdd.BDDNODEZERO:
            rv = inverse
        else:
            row_index = self.uniqid_list.index(node.root)

            row = self.C[row_index]
            rhs = self.d[row_index][0]

            sum_expr = 0.0

            for i, x in enumerate(row):
                if x != 0:
                    var = Real(f'x{i}')

                    if sum_expr == 0:
                        sum_expr = x * var
                    else:
                        sum_expr += x * var

            hi_expr = self.as_z3_expression(node.hi, inverse)
            lo_expr = self.as_z3_expression(node.lo, inverse)

            if hi_expr is False:
                true_side = False
            else:
                true_expr = sum_expr <= rhs

                if hi_expr is True:
                    true_side = true_expr
                else:
                    true_side = And(true_expr, hi_expr)

            if lo_expr is False:
                false_side = False
            else:
                false_expr = sum_expr > rhs

                if lo_expr is True:
                    false_side = false_expr
                else:
                    false_side = And(false_expr, lo_expr)

            # short circuit Or
            if true_side is True or false_side is True:
                rv = True
            elif false_side is False:
                rv = true_side
            elif true_side is False:
                rv = false_side
            else:
                rv = Or(true_side, false_side)

        return rv

    def is_feasible_smt(self, true_rows, false_rows):
        """check if the passed-in constraints are feasible using an smt solver"""

        s = Solver()

        for is_true, rows in ((True, true_rows), (False, false_rows)):
            for row_index in rows:
                row = self.C[row_index]
                rhs = self.d[row_index][0]

                sum_expr = 0.0

                for i, x in enumerate(row):
                    if x != 0:
                        var = Real(f'x{i}')

                        if sum_expr == 0:
                            sum_expr = x * var
                        else:
                            sum_expr += x * var

                if is_true:
                    expr = sum_expr <= rhs
                else:
                    expr = sum_expr > rhs

                s.add(expr)

        for c in s.assertions():
            print(f"Assertion: {c}")

        if s.check() == sat:
            print(f"Sat: {s.model()}\n")
            rv = True
        else:
            print("Unsat\n")
            rv = False

        return rv

    def restrict(self, var_val_dict):
        """
        assign some values to this bdd set variables, updating C and d, and returning the new BDDSet
        """

        bdd = self.bdd.restrict(var_val_dict)

        return BDDset(bdd, self.C, self.d, self.var_list)

    def apply_compose(self, rename_dict, rename_rows):
        """perform in-place compose using the passed-in dict"""

        new_bdd = self.bdd.compose(rename_dict)
        expr = bdd2expr(new_bdd)

        new_var_list = []

        for i, var in enumerate(self.var_list):
            if i in rename_rows:
                new_index = rename_rows[i]
                new_var_list.append(self.var_list[new_index])
            else:
                new_var_list.append(var)

        # replace self (reduces C)
        self.__init__(expr, self.C, self.d, new_var_list)

    def apply_reduce(self):
        """perform in-place reduction from implied constriants"""

        reduced = self._reduce(self.bdd.node, None, [], [])

        if reduced:
            expr = bdd2expr(self.bdd)

            # replace self (reduces C)
            self.__init__(expr, self.C, self.d, self.var_list)

    def _reduce(self, node, parent_node, true_rows, false_rows, use_smt=True):
        """apply reduction due to linear constriants

        returns True if rewired something. Note in this case the bdd's cached
            property 'inputs' may be incorrect. I can't figure out how to clear this.
        """

        assert use_smt, f"use_smt=False unimplemented"
        rv = False

        if node in [boolalg.bdd.BDDNODEONE, boolalg.bdd.BDDNODEZERO]:
            return False

        row_index = self.uniqid_list.index(node.root)

        if parent_node is not None:
            parent_row_index = self.uniqid_list.index(parent_node.root)
        else:
            parent_row_index = None

        print(f"\nrecursive reducing at node {self.var_list[row_index]}")

        # TODO: could probably improve efficiency with a witness

        # try the case where the current constraint is True
        true_rows_temp = true_rows + [row_index]

        if not self.is_feasible_smt(true_rows_temp, false_rows):
            print(f"Rewiring from {self.var_list[parent_row_index]} -> {self.var_list[row_index]} to false child")

            if parent_node.hi == node:
                parent_node.hi = node.lo
            else:
                assert parent_node.lo == node
                parent_node.lo = node.lo

            rv = True

            # recursive case for lo child after rewiring
            self._reduce(node.lo, parent_node, true_rows, false_rows)
        else:
            # hi node is feasible, try lo node
            print(f"hi node was feasible for {self.var_list[row_index]}, trying lo node")
            false_rows_temp = false_rows + [row_index]

            if not self.is_feasible_smt(true_rows, false_rows_temp):
                print(f"Rewiring from {self.var_list[parent_row_index]} -> {self.var_list[row_index]} to true child")
                if parent_node.hi == node:
                    parent_node.hi = node.hi
                else:
                    assert parent_node.lo == node
                    parent_node.lo = node.hi

                rv = True

                # recursive case for hi child after rewiring
                self._reduce(node.hi, parent_node, true_rows, false_rows)
            else:
                print(f"lo node was also feasible for {self.var_list[row_index]}, doing recursive case (hi)")

                rv = self._reduce(node.hi, node, true_rows_temp, false_rows)

                print(f"Doing second recursive case (lo) at node {self.var_list[row_index]}")

                rv = self._reduce(node.lo, node, true_rows, false_rows_temp) or rv

        return rv

    def affineMap(self, A, b):
        """Affine map S * A + b of set S with matrix A and vector b"""

        return self * A + b

    def get_input(self, name):
        """get bdd input by name"""

        rv = None

        for i in self.ordered_inputs:
            if i.name == name:
                rv = i
                break

        if rv is None:
            raise ValueError(name)

        return rv

    def contains(self, pt):
        """ Return True if pt is within the bddset, False o.w. """

        lhs = self.C @ pt
        rhs = np.squeeze(self.d).reshape(lhs.shape)
        assert lhs.shape == rhs.shape, f"{lhs.shape} != {rhs.shape}"
        constraint_booleans = lhs <= rhs

        d = {}
        inputs = self.ordered_inputs

        for i in range(len(constraint_booleans)):
            d[inputs[i]] = constraint_booleans[i]

        rv = self.bdd.restrict(d)
        return rv

    def setdiff(self, set):
        """set difference S1 / S2 of two sets S1 and S2"""

        return self & set.complement()

    def complement(self):
        """complement of a set"""

        return BDDset(bdd2expr(~self.bdd), self.C, self.d)

    def is_equal_smt(self, other):
        """check if two sets are equal using an smt solver approach"""

        return self.is_subset_smt(other) and other.is_subset_smt(self)

    def is_subset_smt(self, other_bddset):
        """check if this set is a subset of another one using an smt solver"""

        rv = True

        # subset if S intersect ~T = emptyset, do this in z3
        # emptyset?(And(S, not(T))

        set_expr = self.as_z3_expression()

        other_complement_expr = other_bddset.as_z3_expression(inverse=True)

        and_expr = And(set_expr, other_complement_expr)

        s = Solver()

        s.add(and_expr)

        if s.check() == sat:
            print(f"S is not a subset of T")
            print(f"S = {set_expr}")
            print(f"~T = {other_complement_expr}")
            print(f"witness in S and ~T: {s.model()}\n")
            rv = False

        return rv


    def verts_list(self, xdim=0, ydim=1, epsilon=1e-6, bounds=(-np.inf, np.inf)):
        """get the vertices projected onto some axes
        this returns a list of lists of 2-d vertices
        """

        rv = []

        if isinstance(xdim, int):
            xdim_vec = np.zeros(self.C.shape[1])
            xdim_vec[xdim] = 1.0

            ydim_vec = np.zeros(self.C.shape[1])
            ydim_vec[ydim] = 1.0
        else:
            assert isinstance(xdim, np.ndarray), f"xdim type was {type(xdim)}, expected int or np.array"
            assert isinstance(ydim, np.ndarray)

            assert xdim.shape == (self.C.shape[1],)
            assert ydim.shape == (self.C.shape[1],)

            xdim_vec = xdim
            ydim_vec = ydim

        plot_tuple = xdim_vec, ydim_vec, bounds, epsilon

        return self._dfs_verts_list(self.bdd.node, ([], []), plot_tuple)

    def _dfs_verts_list(self, node, cur_C_d, plot_tuple):
        """
        do a depth first search to create the list of vertices
        """

        rv = []

        if node is boolalg.bdd.BDDNODEONE:
            verts = _compute_verts(cur_C_d, plot_tuple)

            if verts is not None:
                rv.append(verts)

        elif node is not boolalg.bdd.BDDNODEZERO:
            # recursive case, add constraint based on bdd

            row_index = self.uniqid_list.index(node.root)
            lo_C_d = deepcopy(cur_C_d)
            hi_C_d = cur_C_d # deepcopy(cur_C_d)

            hi_C_d[0].append(self.C[row_index])
            hi_C_d[1].append(self.d[row_index])

            lo_C_d[0].append(-1 * self.C[row_index])
            lo_C_d[1].append(-1 * self.d[row_index])

            rv += self._dfs_verts_list(node.lo, lo_C_d, plot_tuple)
            rv += self._dfs_verts_list(node.hi, hi_C_d, plot_tuple)

        return rv

    def plot(self, color='k', outline_color='k-o', xdim=0, ydim=1, show=True, alpha=1.0, bounds=(-np.inf, np.inf)):
        """plot the set using LP and verts()"""

        verts_list = self.verts_list(xdim, ydim, bounds=bounds)

        for verts in verts_list:
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]

            if outline_color is not None:
                plt.plot(xs, ys, outline_color, alpha=alpha)

            if color is not None:
                plt.fill(xs, ys, color, alpha=alpha)

        if show:
            plt.show()

    def bdd_to_png(self, filename):
        """save bdd to dot format that can be visualized

        needs graphviz: sudo apt install graphviz
        """

        bdd_to_png(self.bdd, filename)

    def unique_var_names(self, other_set):
        """
        return other_set or a copy with no variables names that conflict with self
        """

        rename_dict = {}
        other_var_list = []

        for name in other_set.var_list:
            if name in self.var_list:
                # construct a unique name by adding ticks
                new_name = name + "'"

                while new_name in self.var_list:
                    new_name += "'"

                rename_dict[exprvar(name)] = exprvar(new_name)
                other_var_list.append(new_name)
            else:
                other_var_list.append(name)

        if not rename_dict:
            rv = other_set
        else:
            ex = bdd2expr(other_set.bdd)
            ex_renamed = ex.compose(rename_dict)

            rv = BDDset(ex_renamed, other_set.C, other_set.d, var_list=other_var_list)

        return rv

def _compute_verts(cur_C_d, plot_tuple):
    """recursive base case; returns verts for plotting or None if infeasible"""

    xdim_vec, ydim_vec, bounds, epsilon = plot_tuple

    rv = None

    A_ub = np.array(cur_C_d[0], dtype=float)
    b_ub = np.array(cur_C_d[1], dtype=float)

    # first check feasiblity
    res = linprog(xdim_vec, A_ub, b_ub, bounds=bounds)
    desc = ["normal", "iteration limit reached", "infeasible", "unbounded", "numerical difficulties"]

    if not res.success:
        assert desc[res.status] != 'unbonuded', "Plotting failed because set is unbounded set." + \
            "Use explicit bounds argument to plot with limits."
    else:
        def supp_point_func(vec2d):
            """compute support function given a direction (maximize)"""

            # use negative to maximize
            lpdir = -vec2d[0] * xdim_vec + -vec2d[1] * ydim_vec

            res = linprog(lpdir, A_ub, b_ub, bounds=bounds)

            assert res.success, f"lp failed: {desc[res.status]}"
            assert res.status == 0, f"LP status was nonzero ({res.status}): {desc[res.status]}"

            pt_nd = res.x

            resx = np.dot(xdim_vec, pt_nd)
            resy = np.dot(ydim_vec, pt_nd)

            pt_2d = np.array([resx, resy], dtype=float)

            return pt_2d

        rv = kamenev.get_verts(2, supp_point_func, epsilon=epsilon)

    return rv

def polytope2BDDset(set):
    """convert polytope to BDDset"""

    val = 'a0'

    for i in range(1, set.C.shape[0]):
        val += '& a' + str(i)

    ex = expr(val)

    return BDDset(ex, set.C, set.d)

def bdd_to_png(bdd, filename):
    """save bdd to dot format that can be visualized

    needs graphviz: sudo apt install graphviz
    """

    tmp = tempfile.NamedTemporaryFile(delete=False)

    try:
        s = bdd.to_dot()
        tmp.write(bytes(s, 'utf-8'))

    finally:
        tmp.close()
        check_call(['dot','-Tpng', tmp.name, '-o', filename])
        os.unlink(tmp.name)

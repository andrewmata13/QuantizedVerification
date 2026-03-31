from pyeda.inter import *
import numpy as np
import sys
sys.path.append("..")
from BDDset import BDDset
from sets.Polytope import Polytope

if __name__ == '__main__':

    # construct first BDDset object
    #ex = expr("a & b & c & d")
    varnames = ['x>=0', 'x<=4']
    exprvars = tuple(exprvar(s) for s in varnames)
    ex = And(*exprvars)

    #print(f"ex: {ex}")
    #v2 = exprvar(varnames[1] + "'")
    #ex2 = ex.compose({exprvars[1]: v2})
    #print(f"ex2: {ex2}")
    #exit(1)
    
    C = np.array([[-1], [1]])
    d = np.array([[0], [4]])

    set1 = BDDset(ex, C, d, var_list=varnames)
    #set1.plot('r', alpha=0.25, show=False)

    C = np.array([[-1], [1]])
    d = np.array([[-3], [5]])

    varnames2 = ['x>=3', 'x<=5']
    exprvars2 = tuple(exprvar(s) for s in varnames2)
    ex2 = And(*exprvars2)
    set2 = BDDset(ex2, C, d, var_list=varnames2)
    #set2.plot('b', alpha=0.25, show=False)

    union = set1 | set2

    print(f"num constraints in union set: {union.C.shape[0]}")

    #union.plot(color=None, outline_color='ro', show=False)

    union.bdd_to_png('union.png')

    union.apply_reduce()

    union.bdd_to_png('reduced.png')
    print(f"num constraints in union set after reduction: {union.C.shape[0]}")

    new_varnames = ["x<=5'", "x<=4'"]
    new_bddvars = tuple(bddvar(s) for s in new_varnames)

    # this feels hacky... please improve it
    rename_map = {union.ordered_inputs[1]: new_bddvars[0],
                  union.ordered_inputs[2]: new_bddvars[1]}

    rename_rows = {union.var_list.index('x<=4'): union.var_list.index('x<=5'),
                   union.var_list.index('x<=5'): union.var_list.index('x<=4')}

    union.apply_compose(rename_map, rename_rows)
    union.apply_reduce()

    union.bdd_to_png('reduced2_reordered.png')
    print(f"num constraints in union set after compose: {union.C.shape[0]}")
    
    # construct second BDDset object
    #ex = expr("a & b & c")
    #C = np.array([[-1, 0], [0, -1], [1, 1]])
    #d = np.array([[-1], [-1], [4]])

    #set2 = BDDset(ex, C, d)

    # compute intersection

    #set = set1 | set2

    # visualize the resulting set
    #set.plot('g')

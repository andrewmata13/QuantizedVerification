from pyeda.inter import *
import numpy as np
import sys
sys.path.append("..")
from BDDset import BDDset, bdd_to_png
from sets.Polytope import Polytope

def main():
    """main entry point"""
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

    union = set1 | set2

    union.bdd_to_png('S_orig.png')

    d = {union.get_input('x<=4'): True}
    true_set = union.restrict(d)
    true_set.bdd_to_png('Sc_true.png')
    print(f"Sc_true: {sample_set(true_set, -2, 7)}")
    print(f"Sc_true variables: {true_set.var_list}")

    d = {union.get_input('x<=4'): False}
    false_set = union.restrict(d)
    false_set.bdd_to_png('Sc_false.png')
    print(f"Sc_false: {sample_set(false_set, -2, 7)}")
    print(f"Sc_false variables: {false_set.var_list}")

    either_set = true_set.constraint_preserving_union(false_set)
    either_set.bdd_to_png('Sc_either.png')
    print(f"Sc_either: {sample_set(either_set, -2, 7)}")
    print(f"Sc_eiter variables: {either_set.var_list}")

def sample_set(bdd, lb, ub, step=0.5):
    """sample a bdd from lb to ub and print result"""

    s = ""
    num = 1 + (ub - lb) / step

    for i in np.linspace(lb, ub, int(num)):
        s1 = i-0.1
        in1 = bdd.contains([s1])
        
        s2 = i+0.1
        in2 = bdd.contains([s2])

        if in1 and in2:
            s += "*"
        elif not in1 and not in2:
            s += " "
        else:
            s += f"{int(i)}"

    return s

if __name__ == '__main__':
    main()


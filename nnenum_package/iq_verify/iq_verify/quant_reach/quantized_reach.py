import abc
import numpy as np
import copy
import scipy
import itertools
import math

from iq_verify.quant_reach.quantization_utils import QuantizationUtils
from iq_verify.set_repr import h_polytope, star_set, bounding_box, point_repr

#COLORS = itertools.cycle([ "red", "blue", "lime", "orange", "magenta", "cyan", "green", "grey", "darkkhaki", "silver", "darkgrey", "mediumslateblue", "limegreen", "palegreen", "sienna", "peachpuff", "saddlebrown", "cadetblue", "cornflowerblue", "indigo", "palevioletred", "lightslategray", "salmon", "fuchsia", "pink", "darkolivegreen", "navy", "hotpink", "darksalmon", "lightblue", "crimson", "aquamarine", "goldenrod", "olivedrab", "gainsboro", "sandybrown", "mediumvioletred", "powderblue", "gray", "tan", "lightseagreen", "khaki", "deepskyblue", "mediumaquamarine", "springgreen", "coral", "darkturquoise", "darkgreen", "slategrey", "rebeccapurple", "thistle", "darkviolet", "darkgoldenrod", "blanchedalmond", "orangered", "purple", "mediumturquoise", "darkgray", "turquoise", "mediumblue", "lawngreen", "darkseagreen", "darkred", "darkmagenta", "brown", "firebrick", "darkslateblue", "palegoldenrod", "lightgreen", "steelblue", "chartreuse", "aqua", "midnightblue", "blueviolet", "tomato", "lightsalmon", "slateblue", "lightslategrey", "darkorange", "mediumspringgreen", "mediumpurple", "dodgerblue", "lavender", "skyblue", "wheat", "lightsteelblue", "lightcoral", "forestgreen", "moccasin", "burlywood", "orchid", "violet", "dimgrey", "darkcyan", "mediumorchid", "olive", "lightskyblue", "darkslategray", "plum", "mediumseagreen", "chocolate", "paleturquoise", "seagreen", "deeppink", "darkblue", "lightgrey", "darkorchid", "maroon", "gold", "lightcyan", "dimgray", "rosybrown", "peru", "slategray", "indianred", "lightgray", "lightpink", "teal", "darkslategrey", "bisque", "royalblue", "papayawhip" ])
COLORS = itertools.cycle(["blue"])

DIMS_TO_PLOT = (0, 1)

CMD_SEQ = "CMD_SEQ"


def set_dims_to_plot(dims):
    global DIMS_TO_PLOT
    DIMS_TO_PLOT = copy.deepcopy(dims)
    # TODO: Move DIMS_TO_PLOT to set_repr and have all other subclasses inherit
    # it, so that we only need to set it once.
    #
    # In fact, shouldn't this entire function just be moved to set_repr?
    h_polytope.DIMS_TO_PLOT = copy.deepcopy(dims)
    star_set.DIMS_TO_PLOT = copy.deepcopy(dims)
    bounding_box.DIMS_TO_PLOT = copy.deepcopy(dims)
    point_repr.DIMS_TO_PLOT = copy.deepcopy(dims)


class Counterexample:
    def __init__(self, init_pt, init_subset, unsafe_subset, cmd_seq):
        self.init_pt = init_pt
        self.init_subset = init_subset
        self.unsafe_subset = unsafe_subset
        self.cmd_seq = cmd_seq


class QuantizedReach(abc.ABC):
    def __init__(
        self,
        n_dims,
        dt,
        initial_sets,
        quant_params,
        state_to_cmd,
        get_dynamics,
        get_offset,
        optimize_with_rle,
        safe_sets=None,
        unsafe_sets=None,
        show_plot=False,
    ):
        self.n_dims = n_dims
        self.dt = dt

        self.initial_sets = initial_sets

        self.unsafe_sets = unsafe_sets
        if self.unsafe_sets is None:
            self.unsafe_sets = []

        self.safe_sets = safe_sets
        if self.safe_sets is None:
            self.safe_sets = []

        # The side-lengths of the N-dimensional cells into which the starting
        # states will be split.
        assert isinstance(quant_params, list) or isinstance(quant_params, np.ndarray), "Quantization parameters must be provided as a list/vector"
        #####assert len(quant_params) == n_dims, "Must provide quantization parameters for each dimension"
        self.quant_params = quant_params

        # function that maps a single state to a command
        self.state_to_cmd = state_to_cmd

        # function that takes a command, and returns the dynamics matrix
        self.get_dynamics = get_dynamics

        # function that takes a command, and returns the offset vector
        self.get_offset = get_offset

        # whether or not to plot statesets
        self.show_plot = show_plot

        # whether or not to optimize the reachability step by grouping together
        # quantized subsets that share the same control action
        self.optimize_with_rle = optimize_with_rle

        # the number of times the quantization parameters have been refined;
        # used to determine which parameter to halve next.
        self.refinement_count = 0


    @abc.abstractmethod
    def plot_initial_safe_unsafe(self):
        return

    @abc.abstractmethod
    def _check_state(self, stateset):
        return

    @abc.abstractmethod
    def find_counterexample(self, stateset):
        return

    @abc.abstractmethod
    def verify_quantized_safety(self):
        return


    def apply_cmd_forward(self, stateset, cmd, in_place=True):
        A = self.get_dynamics(cmd)

        c = self.get_offset(cmd=cmd)
        BV = (np.eye(self.n_dims)*self.dt + (A * self.dt**2 / math.factorial(2)) + (A**2 * self.dt**3 / math.factorial(3)) + (A**3 * self.dt**4 / math.factorial(4)) + (A**4 * self.dt**5 / math.factorial(5))) @ c

        # Apply dynamics.
        # When doing forward reachability, we can modify the stateset object
        # in-place; we do not need to retain a copy of its current position.
        e_At = scipy.linalg.expm(A * self.dt)
        stateset = stateset.affine_transform(e_At, BV, in_place=in_place)
        stateset.properties[CMD_SEQ] += [cmd]
        return stateset


    def apply_cmd_backward(self, stateset, cmd, in_place=True):
        # Obtain backward dynamics
        A = self.get_dynamics(cmd)

        c = self.get_offset(cmd=cmd)
        BV = (np.eye(self.n_dims)*self.dt + (A * self.dt**2 / math.factorial(2)) + (A**2 * self.dt**3 / math.factorial(3)) + (A**3 * self.dt**4 / math.factorial(4)) + (A**4 * self.dt**5 / math.factorial(5))) @ c

        # Apply backward dynamics.
        e_neg_At = scipy.linalg.expm(-1 * A * self.dt)

        # NOTE: When performing an affine transformation forward, the linear
        # transformation occurs before the offset is applied. So when we step
        # backward, we must apply the backward offset before the backward
        # linear transformation, to achieve the reversed operation.
        #
        # This operation CANNOT be performed in-place; we still need a copy of
        # the stateset in its current position, so that we can back-propagate
        # using the other actions.
        new_stateset = stateset.affine_transform(np.eye(stateset.n_dims), -BV, in_place=False)
        new_stateset = new_stateset.affine_transform(e_neg_At, np.zeros((new_stateset.n_dims, 1)), in_place=in_place)

        new_stateset.properties[CMD_SEQ] = [cmd] + copy.deepcopy(stateset.properties[CMD_SEQ])

        return new_stateset


    def refine_quant_params(self):
        """
        Default behavior: Rotate through the quantization parameters
        """
        self.quant_params[self.refinement_count % len(self.quant_params)] /= 2
        self.refinement_count += 1


    def trim_safe_portions(self, stateset):
        """
        Default implementation only considers continuous state
        """
        statesets = [stateset]

        safe_sets = self.safe_sets

        for safe_set in safe_sets:
            new_statesets = []
            for s in statesets:
                new_statesets += s.set_difference(safe_set)
            statesets = new_statesets

        return statesets


    def split_stateset(self, stateset, assumed_cmd=None):
        quant_utils = QuantizationUtils(
            quant_params=self.quant_params,
            q_to_cmd=self.state_to_cmd,
            optimize_with_rle=self.optimize_with_rle,
            only_construct_T_for_desired_cmd=assumed_cmd,
        )
        qs_Qs_and_cmds = quant_utils.split_stateset(stateset=stateset)
        return qs_Qs_and_cmds

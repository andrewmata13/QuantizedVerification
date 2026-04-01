import copy
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy
import itertools
import logging
log = logging.getLogger(__name__)
import gurobipy

from iq_verify.quant_reach import quantized_reach
from iq_verify.set_repr import set_repr, h_polytope
from iq_verify.quant_reach.quantized_reach import COLORS, CMD_SEQ, Counterexample


class QuantizedBackwardReach(quantized_reach.QuantizedReach):
    def __init__(self, *args, possible_commands=None, **kwargs):
        super().__init__(*args, **kwargs)

        # A discrete list for backward reachability of commands that the controller could issue.
        self.possible_commands = possible_commands
        assert self.possible_commands is not None

    def plot_initial_safe_unsafe(self):
        if not self.show_plot:
            return

        # Plot the unsafe, safe, and initial sets.
        for safe in self.safe_sets:
            safe.plot("lime", fill=True)
        for unsafe in self.unsafe_sets:
            unsafe.plot("red", fill=True)
        for initial in self.initial_sets:
            initial.plot("cyan", fill=True)

        unsafe_patch = mpatches.Patch(color="cyan", alpha=0.5, label="Unsafe Set (backreach begins here)")
        initial_patch = mpatches.Patch(color="red", alpha=0.5, label="Initial Set (reached => unsafe trajectory exists)")
        safe_patch = mpatches.Patch(color="lime", alpha=0.5, label="Safe Set (reached => okay to discard safe portion)")

        plt.legend(handles=[unsafe_patch, initial_patch, safe_patch], fontsize=12, loc="upper left")


    def verify_quantized_safety(self):
        """
        Perform backward reachability. Return safe, or unsafe plus counterexample.
        """
        # Each possible predecessor command will be considered.
        assert self.possible_commands is not None, "Must provide list of possible_commands"

        for unsafe_set in self.unsafe_sets:
            if not isinstance(unsafe_set, set_repr.SetRepr):
                raise ValueError(f"Unsupported set representation {type(unsafe_set)} for unsafe_set to perform backward reachability")

            unsafe_set.properties[CMD_SEQ] = []

        for unsafe_set in self.unsafe_sets:
            counterexample = self._check_state(unsafe_set)
            if counterexample is not None:
                return counterexample

        return None


    def _check_state(self, starting_set):
        S_list = [starting_set]
        popped = 0
        n_deadends = 0
        while S_list:
            if popped % 100 == 0:
                lens = [len(s.properties[CMD_SEQ]) for s in S_list]
                log.info(f"popped {popped}, unique_paths: {n_deadends}, remaining_work: {len(S_list)}, max_len: {max(lens)}")

            S = S_list.pop()
            popped += 1

            if S.check_is_feasible() == False:
                continue

            if self.show_plot:
                # TODO: Should we add a "color" field, and plot the entire reachability plot in a single color?
                S.plot(next(COLORS))

            # search for intersection with unsafe
            counterexample = self.find_counterexample(S)
            if counterexample is not None:
                # counterexample was found (TODO: return multiple, if present)
                # TODO: Return the remainder of the S_list, so that we can pick up where we left off (only matters with falsification enabled).
                return counterexample

            # Assume each of the commands was chosen to reach the current state.
            # We will only consider predecessor states that issue this assumed
            # command.
            predecessors = []
            for assumed_cmd in self.possible_commands:
                # apply backward dynamics
                P = self.apply_cmd_backward(
                    stateset=S,
                    cmd=assumed_cmd,
                )

                # Split P by command along quantization boundaries..
                # Only keep the partitions whose quantized point maps to the assumed command.
                # TODO: Test how many HPoly constructions we avoid by including "assumed_cmd".
                qs_Ps_cmds = self.split_stateset(stateset=P, assumed_cmd=assumed_cmd)
                Ps = [P for (q, P, c) in qs_Ps_cmds if c == assumed_cmd]

                # Trim away safe portions of transformed set
                Ts = []
                for P in Ps:
                    Ts += self.trim_safe_portions(stateset=P)

                # Prepend the transformed, trimmed sets to the S_list
                predecessors += Ts

            # TODO: This can have a significant impact on the search order,
            #       as well as the length of the S_list.
            #predecessors.reverse()

            if len(predecessors) == 0:
                n_deadends += 1

            S_list = S_list + predecessors

        # No counterexample was found.
        return None


    def find_counterexample(self, stateset):
        """
        Default implementation only considers continuous state

        Returns: (
            Witness point in the initial set, to be simulated for replay
            Subset of the initial set that is reached
            Subset of the unsafe set that reaches the initial set by following the reversed CMD_SEQ
        )
        """
        # quick check if intersection even exists
        init_counterexamples = []
        for initial_set in self.initial_sets:
            # quick check if intersection even exists
            if not stateset.check_for_intersection(initial_set):
                continue

            # find the exact intersection
            subset_of_init = stateset.intersect(initial_set)
            if not subset_of_init or not subset_of_init.check_is_feasible():
                continue

            # retrace subset_of_unsafe back to initial set
            subset_of_unsafe = subset_of_init.undo_all_transformations()
            cheb_radius, init_pt = subset_of_init.get_chebyshev_center()
            init_counterexamples.append(init_pt)

            if self.show_plot:
                stateset.plot(next(COLORS))
                subset_of_init.plot("black")
                subset_of_unsafe.plot("black")

        if init_counterexamples:
            init_pt = init_counterexamples[0] # TODO: Return ALL unsafe witnesses
            cmd_seq = stateset.properties[CMD_SEQ]
            rv = Counterexample(
                init_pt=init_pt,
                init_subset=subset_of_init,
                unsafe_subset=subset_of_unsafe,
                cmd_seq=cmd_seq,
            )
            return rv

        # No unsafe intersections found
        return None

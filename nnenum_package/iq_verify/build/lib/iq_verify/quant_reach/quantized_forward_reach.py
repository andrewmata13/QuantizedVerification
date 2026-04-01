import abc
import copy
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy
import itertools
import logging
log = logging.getLogger(__name__)

from iq_verify.quant_reach import quantized_reach
from iq_verify.quant_reach.quantization_utils import QuantizationUtils
from iq_verify.quant_reach.quantized_reach import COLORS, CMD_SEQ, Counterexample
from iq_verify.set_repr import set_repr, point_repr, star_set, h_polytope, bounding_box, aggregation_utils


class QuantizedForwardReach(quantized_reach.QuantizedReach):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def plot_initial_safe_unsafe(self):
        if not self.show_plot:
            return

        for safe in self.safe_sets:
            safe.plot("lime", fill=True)
        for initial in self.initial_sets:
            initial.plot("cyan", fill=True)
        for unsafe in self.unsafe_sets:
            unsafe.plot("red", fill=True)

        initial_patch = mpatches.Patch(color='cyan', alpha=0.5, label='Initial Set (foward reach begins here)')
        unsafe_patch = mpatches.Patch(color='red', alpha=0.5, label='Unsafe Set (reached => quantized system unsafe)')
        safe_patch = mpatches.Patch(color='lime', alpha=0.5, label='Safe Set (reached => okay to discard safe portion)')

        plt.legend(handles=[initial_patch, unsafe_patch, safe_patch], fontsize=12, loc="upper left")


    def verify_quantized_safety(self):
        """
        Perform forward reachability using quantized points for control inputs,
        up to the given simulation end time.
        """
        for initial in self.initial_sets:
            if not isinstance(initial, set_repr.SetRepr):
                raise ValueError(f"Unsupported set representation {type(initial)} for initial set to perform forward reachability")
            initial.properties[CMD_SEQ] = []

        for initial in self.initial_sets:
            counterexample = self._check_state(initial)
            if counterexample is not None:
                return counterexample

        return None


    def _check_state(self, starting_set):
        S_list = [starting_set]
        popped = 0
        while S_list:
            if popped % 100 == 0:
                lens = [len(s.properties[CMD_SEQ]) for s in S_list]
                log.info(f"popped {popped}, remaining_work: {len(S_list)}, max_len: {max(lens)}")

            S = S_list.pop()
            popped += 1

            if self.show_plot:
                S.plot(next(COLORS), fill=False)

            # Split S along quantization boundaries by command.
            qs_Ss_cmds = self.split_stateset(stateset=S)

            # Transform each partition according to its command.
            Ps = []
            for _, Sq, cmd in qs_Ss_cmds:
                P = self.apply_cmd_forward(stateset=Sq, cmd=cmd)
                Ps.append(P)

            # Process the transformed sets.
            Ts = []
            for P in Ps:
                if not P.check_is_feasible():
                    continue

                # search for intersection with unsafe
                counterexample = self.find_counterexample(P)
                if counterexample is not None:
                    print("# OF SETS:", popped)
                    return counterexample

                # Trim away safe portions of transformed set
                Ts += self.trim_safe_portions(stateset=P)

            # TODO: Aggregate propagated reachable sets into one set
            AGGREGATE_SETS = False
            if AGGREGATE_SETS and len(Ts) > 1:
                log.info(f"aggregating {len(Ts)} sets... ")
                overapprox = aggregation_utils.AggregationUtils.aggregate_into_pca_StarSet(Ts, self.n_dims)
                Ts = [overapprox]
                log.info(f"done")

            # Add the transformed, trimmed sets to the S_list
            S_list += Ts

        # did not find any counterexample
        print("# OF SETS:", popped)
        return None

    def find_counterexample(self, stateset):
        """
        Default implementation only considers continuous state

        Returns: (
            Witness point in the initial set, to be simulated for replay
            Subset of the initial set that reaches the unsafe set by following the CMD_SEQ in order
            Subset of the unsafe set that is reached
        )
        """
        init_counterexamples = []
        for unsafe_set in self.unsafe_sets:
            # quick check if intersection even exists
            if not stateset.check_for_intersection(unsafe_set):
                continue

            # find the exact intersection
            subset_of_unsafe = stateset.intersect(unsafe_set)
            if not subset_of_unsafe or not subset_of_unsafe.check_is_feasible():
                continue

            # retrace subset_of_unsafe back to initial set
            subset_of_init = subset_of_unsafe.undo_all_transformations()
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


    def simulate_point(self, point, quantized, t_end=np.inf, color=None, alpha=1.0, plot_dim0_over_time=False):
        """
        point - a set_repr.point_repr.Point
        quantized - whether or not the state should be quantized before being passed to the neural network

        returns: (is_safe, cmd_seq)
        """
        # TODO: Set these dims in a better place; right now, we need to keep setting and re-setting them.
        from iq_verify.quant_reach.quantized_reach import DIMS_TO_PLOT
        init_pt = point_repr.Point(
            n_dims=self.n_dims,
            coords=point,
            discrete=copy.deepcopy(self.initial_sets[0].properties),
        )
        init_pt.properties[CMD_SEQ] = []

        points = [init_pt]
        def plot_sim():
            pt_color = color if color is not None else next(COLORS)
            if plot_dim0_over_time:
                plt.plot([time for time in times], [p.coords[DIMS_TO_PLOT[0], :] for p in points], color=pt_color, marker=".", markersize=3, alpha=alpha)
            else:
                plt.plot([p.coords[DIMS_TO_PLOT[0], :] for p in points], [p.coords[DIMS_TO_PLOT[1], :] for p in points], color=pt_color, marker=".", markersize=3, alpha=alpha)

        t = 0
        times = [t]
        while t <= t_end:
            t += self.dt
            times.append(t)
            p = points[-1]

            if quantized == True:
                # use quantized version of coordinates
                # TODO: Pass Point, not coords, so that discrete dict can be
                # used. Or provide an optional arg for quantize_point so that
                # overriding implementations can use info besides coordinates.
                q = QuantizationUtils.quantize_point(p.coords, self.quant_params)
                cmd = self.state_to_cmd(q)
            else:
                # use point directly
                cmd = self.state_to_cmd(p.coords)

            # Propagate point forward by one timestep.
            # Do NOT perform the operation in-place, as we may need to plot/print the list of all points visited.
            p_new = self.apply_cmd_forward(p, cmd, in_place=False)

            assert (p.coords.shape == p_new.coords.shape), f"shapes {p.coords.shape} and {p_new.coords.shape} do not match"
            points.append(p_new)

            counterexample = self.find_counterexample(p_new)

            if counterexample is not None:
                # Reached unsafe set
                sim_is_safe = False
                sim_cmd_seq = p_new.properties[CMD_SEQ]
                log.debug(f"Simulation reached unsafe set at t={t}")
                if self.show_plot:
                    plot_sim()
                return sim_is_safe, sim_cmd_seq

            if not self.trim_safe_portions(stateset=p_new):
                # Reached explicitly safe set
                sim_is_safe = True
                sim_cmd_seq = p_new.properties[CMD_SEQ]
                log.debug(f"Simulation reached safe set at t={t}")
                if self.show_plot:
                    plot_sim()
                return sim_is_safe, sim_cmd_seq

        log.debug("Simulation reached t_end without reaching either safe set or unsafe set")
        if self.show_plot:
            plot_sim()
        return None, None

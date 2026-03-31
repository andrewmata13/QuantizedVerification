import numpy as np
import itertools
import operator

from iq_verify.set_repr import h_polytope

UNIQUE_GID = itertools.count(1)


class qData:
    def __init__(this_qData, q, cmd, group):
        this_qData.q = q
        this_qData.cmd = cmd
        this_qData.group = group
    def __str__(this_qData):
        return f"qData: q:{this_qData.q}, cmd:{this_qData.cmd}, group:{this_qData.group}"
    def __repr__(this_qData):
        return this_qData.__str__()


class QuantizationUtils:
    """
    Utility class to help with quantizing points, splitting stateset along
    quantization boundaries, etc.
    """
    def __init__(self,
        quant_params,
        q_to_cmd,
        optimize_with_rle,
        only_construct_T_for_desired_cmd=None,
        check_command_equality=operator.eq,
    ):
        self.quant_params = quant_params
        self.q_to_cmd = q_to_cmd
        self.optimize_with_rle = optimize_with_rle
        self.only_construct_T_for_desired_cmd = only_construct_T_for_desired_cmd
        self.check_command_equality = check_command_equality


    @staticmethod
    def quantize_point(point, quant_params):
        """
        Map a single state to its quantized state with respect to the given quantization parameters.
        """
        pt = np.array(point)
        n_dims = len(pt)

        # Temporary placeholder, to be filled in at each dimension.
        qpoint = np.empty(n_dims)

        for d in range(n_dims):
            # round down [[[
            # Reference: https://stackoverflow.com/a/70210770/20020749
            num = pt[d]
            nearest = np.round(num / quant_params[d]) * quant_params[d]

            if np.isclose(nearest, num):
                qpoint[d] = num
            elif nearest < num:
                qpoint[d] = nearest
            else:
                qpoint[d] = nearest - quant_params[d]
            # ]]] round down

            # So far, we have found the lower boundary of this quantization cell.
            # What we really want is the center of that cell, which serves as
            # the quantization point for control input calculation. Therefore,
            # we must look at the halfway mark between the left and right
            # cell boundaries in this dimension.
            qpoint[d] += (quant_params[d] / 2)

        rv = np.reshape(qpoint, (n_dims, 1))
        return rv


    @staticmethod
    def stateset_to_qpoints(stateset, quant_params):
        """
        Default implementation: determine quantized points based on axis-aligned quantization grid.
        """
        n_dims = stateset.n_dims

        box = stateset.get_bounding_box()

        nd_marks = []
        for d in range(n_dims):
            # lb refers to the first quantization boundary to the right of the
            # stateset's leftmost point. In other words, it serves as the first
            # mark on which the stateset is split.
            #
            # ub refers to the first quantization boundary that lies to the right
            # of the stateset's rightmost point. It will always be outside the
            # stateset.
            #
            # Note: Rounding the lower and upper bounds helps to prevent a
            # floating-point error. The error that we were seeing was, that the
            # lower bound would have some additional floating point value such
            # as 1.000000000000000004, but the upper bound would be exactly 1.0.
            # When combined with the floor/ceil operations involved with
            # quantization, this led to lb < ub, which causes issues.
            lower = round(box.l(d), 6)
            upper = round(box.u(d), 6)
            lb = lower // quant_params[d] * quant_params[d] + quant_params[d]
            ub = upper // quant_params[d] * quant_params[d] + quant_params[d]

            d_marks = []

            mark = lb
            while mark < ub + 1e-6:
                # "mark" is the beginning of the quantized region; we want the
                # qpoint to be the center.
                d_marks.append(mark - quant_params[d]/2)
                mark += quant_params[d]
            nd_marks.append(d_marks)

        pts = [np.array(t) for t in itertools.product(*nd_marks)]
        return pts

    def q_to_T(self, q):
        """
        Default implementation returns axis-aligned rectangle HPoly centered at q
        """
        n_dims = len(q)
        quant_params = self.quant_params

        lhs_list = []
        rhs_list = []

        for d in range(n_dims):
            lo = q[d] - quant_params[d]/2
            hi = q[d] + quant_params[d]/2

            # gte lo
            d_lhs = [0] * n_dims
            d_lhs[d] = -1.0
            lhs_list.append(d_lhs)
            rhs_list.append(-lo)

            # lte hi
            d_lhs = [0] * n_dims
            d_lhs[d] = 1.0
            lhs_list.append(d_lhs)
            rhs_list.append(hi)

        rv = h_polytope.HPoly(
            n_dims=n_dims,
            LHS=np.array(lhs_list),
            rhs=np.array(rhs_list)
        )
        return rv

    @staticmethod
    def split_P(P, qs_Ts_cmds):
        qs_Qs_and_cmds = []
        for q, T, c in qs_Ts_cmds:
            if T is None:
                continue
            Q = P.intersect(T)
            qs_Qs_and_cmds.append((q, Q, c))
        return qs_Qs_and_cmds


    def split_stateset(self, stateset):
        """
        Split the stateset along the given quantization grid

        Update 2025-07-24: Return a three-tuple:
            - qs, List of quantized cell centers
            - Ts, List of sets INTERSECT(cell(q), stateset) for q in qs
            - cs, List of commands GET_CMD(q) for q in qs
        """
        q_list = self.stateset_to_qpoints(stateset, self.quant_params)
        assert len(q_list) > 0
        n_dims = len(q_list[0])

        cmds_list = []
        for q in q_list:
            c = self.q_to_cmd(q)
            cmds_list.append(c)


        if self.optimize_with_rle:
            if len(set(cmds_list)) == 1:
                # best optimization, no splitting at all
                return [(q_list[0], stateset, cmds_list[0])]

            # some optimization -- try grouping points q into rectangles, similar to run-length encoding
            qs_Ts_cmds = self.qs_to_Ts_with_RLE(n_dims, q_list, stateset)

        else:
            # no optimization -- intersect the stateset with the cell T of each q
            qs_Ts_cmds = []
            for q, c in zip(q_list, cmds_list):
                if (self.only_construct_T_for_desired_cmd is not None) and qDatas[0].cmd != self.only_construct_T_for_desired_cmd:
                    continue
                T = self.q_to_T(q)
                qs_Ts_cmds.append((q,T,c))

        qs_Qs_and_cmds = self.split_P(stateset, qs_Ts_cmds)
        return qs_Qs_and_cmds


    @staticmethod
    def prod(shape, axes):
        """
        Similar to itertools.product, but with "axes" to specify the order of iteration.

        Reference - https://stackoverflow.com/a/9969179/20020749
        """
        prod_T = tuple(zip(*itertools.product(*(range(shape[axis]) for axis in axes))))

        prod_T_ordered = [None] * len(axes)
        for i, axis in enumerate(axes):
            prod_T_ordered[axis] = prod_T[i]
        return zip(*prod_T_ordered)

    @staticmethod
    def get_xyz_coords(arrND):
        n_dims = len(arrND.shape)
        return list(QuantizationUtils.prod(arrND.shape, range(n_dims-1, -1, -1)))


    def q_list_to_q_grid(self, q_list):
        # Starting list; may have irregular shape
        assert len(q_list) > 0
        q_arr = np.array(q_list)
        n_dims = len(q_list[0])

        grid = []
        lens = []
        for d in range(n_dims):
            ds = np.unique(q_arr[:, d]).tolist()
            grid.append(ds)
            lens.append(len(ds))

        # By this point, the list is rectangular in N dimensions
        rect = np.array(list(itertools.product(*grid)))

        arrND = np.full(shape=lens, fill_value=None, dtype=object)

        coords = list(itertools.product(*[range(l) for l in lens]))

        for i, coord in enumerate(coords):
            val = rect[i]
            qdat = qData(q=val, cmd=self.q_to_cmd(val), group=None)
            arrND[coord] = qdat

        return arrND

    def qs_to_Ts_with_RLE(self, n_dims, q_list, stateset):
        def find_first_ungrouped(arrND):
            coords = QuantizationUtils.get_xyz_coords(arrND)
            for c in coords:
                if arrND[c].group is None:
                    return c
            # all elements have been grouped; there is no leader
            return None

        def group_from_leader(arrND, leader):
            coords = QuantizationUtils.get_xyz_coords(arrND)
            maxes = arrND.shape

            class IVal:
                def __init__(self, lo, hi):
                    self.lo = lo
                    self.hi = hi

                def elems(self):
                    return [i for i in range(self.lo, self.hi+1)]

                def update(self, elems):
                    self.lo = min(elems)
                    self.hi = max(elems)

                def __str__(self):
                    return f"[{self.lo} -- {self.hi}]"

                def __repr__(self):
                    return self.__str__()

            ivals = {}
            for d in range(n_dims):
                ivals[d] = IVal(lo=leader[d], hi=maxes[d] - 1)

            for d in range(n_dims):
                lower_dims = list(range(0, d))
                higher_dims = list(range(d+1, n_dims))

                lowers = [ ivals[l].elems() for l in lower_dims ]
                prepend = [list(tup) for tup in itertools.product(*lowers)]

                middles = [k for k in ivals[d].elems()]

                highers = [ ivals[h].elems() for h in higher_dims ]
                postpend = [list(tup) for tup in itertools.product(*highers)]

                # Figure out which ones from this dimension we can keep
                keepers = []
                def get_keepers():
                    for post in postpend:
                        for mid in middles:
                            for pre in prepend:
                                c = tuple(pre + [mid] + post)
                                if not self.check_command_equality(arrND[c].cmd, arrND[leader].cmd):
                                    return
                            keepers.append(mid)
                get_keepers()
                ivals[d].update(keepers)


            listoflists = [ivals[d].elems() for d in range(n_dims)]
            c_of_leader_group = list(itertools.product(*listoflists))
            leader_gid = next(UNIQUE_GID)
            for c in c_of_leader_group:
                arrND[c].group = leader_gid
            # return the inputted arrND, but with the .group field assigned in all the
            # elements of the rectangle cornerstoned by the leader.
            return arrND

        """
        Start of function
        """
        arrND = self.q_list_to_q_grid(q_list)
        leader = find_first_ungrouped(arrND)
        while leader is not None:
            arrND = group_from_leader(arrND, leader)
            leader = find_first_ungrouped(arrND)

        coords = QuantizationUtils.get_xyz_coords(arrND)

        gid_to_qDatas = {}
        for c in coords:
            qd = arrND[c]
            gid = qd.group
            if gid not in gid_to_qDatas.keys():
                gid_to_qDatas[gid] = []
            gid_to_qDatas[gid].append(qd)

        qs_Ts_cmds = []
        for gid, qDatas in gid_to_qDatas.items():
            if (self.only_construct_T_for_desired_cmd is not None) and qDatas[0].cmd != self.only_construct_T_for_desired_cmd:
                continue
            T = self.qDatas_to_T(qDatas)
            # arbitrarily return first q in list
            qs_Ts_cmds.append((qDatas[0].q, T, qDatas[0].cmd))

        return qs_Ts_cmds


    def qDatas_to_T(self, qDatas):
        qs = [qd.q for qd in qDatas]
        n_dims = len(qs[0])

        mins = np.array([min([ q[d] for q in qs ]) for d in range(n_dims)])
        maxes = np.array([max([ q[d] for q in qs ]) for d in range(n_dims)])

        lo = mins - (np.array(self.quant_params) / 2)
        hi = maxes + (np.array(self.quant_params) / 2)

        rv = h_polytope.HPoly(
            n_dims=n_dims,
            LHS=np.vstack((
                -np.eye(n_dims),
                np.eye(n_dims),
            )),
            rhs=np.append(-lo, hi),
        )
        return rv

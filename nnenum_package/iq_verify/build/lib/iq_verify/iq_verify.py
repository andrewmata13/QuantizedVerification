import logging
log = logging.getLogger(__name__)

from iq_verify.quant_reach import quantized_reach, quantized_forward_reach, quantized_backward_reach


class IQVerify:
    def __init__(
            self,
            n_dims,
            dt,
            quant_params,
            state_to_cmd,
            get_dynamics,
            get_offset,
            initial_sets=None,
            safe_sets=None,
            unsafe_sets=None,
            possible_commands=None,
            show_plot=False,
            falsify_nonquantized_system=False,
            optimize_with_rle=True,
    ):
        # Number of dimensions in the state space. Assumed to also be the
        # dimension of the neural network inputs, though these are allowed to
        # be different with a little bit of extra setup.
        self.n_dims = n_dims

        # Timestep size between reachability steps (and simulation steps).
        self.dt = dt

        # An n-dimensional list of quantization parameters. By default, the
        # state space is gridded in an axis-aligned manner.
        self.quant_params = quant_params

        # Mapping from n-dimensional state to a command. Typically, this means
        # passing the state to a neural network.
        self.state_to_cmd = state_to_cmd

        # Mapping from a command to the dynamics matrix for one timestep.
        self.get_dynamics = get_dynamics

        # Mapping from a command to the offset vector for one timestep.
        self.get_offset = get_offset

        # The list of initial statesets considered for reachability.
        self.initial_sets = initial_sets

        # The list of safe or invariant sets that indicate safe termination.
        # States that reach these sets are no longer considered in reachability.
        self.safe_sets = safe_sets

        # The list of unsafe sets that indicate unsafe termination.
        self.unsafe_sets = unsafe_sets

        # Backward reachability only -- The discrete list of commands that the
        # controller can issue.
        self.possible_commands = possible_commands

        # Whether or not to plot the reachability result.
        self.show_plot = show_plot

        # Whether or not to perform stateset regrouping, similar to run-length encoding.
        self.optimize_with_rle = optimize_with_rle

        # Whether or not to refine quantization parameters in search of a
        # falsifying counterexample to the original non-quantized system.
        #
        # If True, quantization parameters will be refined until either the
        # quantized system is verified safe, or an unsafe counterexample to the
        # original non-quantized system is found.
        #
        # If False, only the user-provided quant_params are considered.
        # That is, "The quantized system is unsafe" is an acceptable result.
        self.falsify_nonquantized_system = falsify_nonquantized_system

        # The quantized reachability objects used to perform verification.
        self.forward = None
        self.backward = None

        # The constructor methods for the quantized reachability objects.
        # Users can override these default classes with implementations specific
        # to their use-case.
        self.forward_class = quantized_forward_reach.QuantizedForwardReach
        self.backward_class = quantized_backward_reach.QuantizedBackwardReach

        # Whether or not the quantized reachability objects have been initialized.
        self.forward_initialized = False
        self.backward_initialized = False


    def _initialize_forward_reach(self):
        """
        Initializes the forward reach object using the specified quantized reachability class.
        """
        self.forward = self.forward_class(
            n_dims=self.n_dims,
            dt=self.dt,
            quant_params=self.quant_params,
            state_to_cmd=self.state_to_cmd,
            get_dynamics=self.get_dynamics,
            get_offset=self.get_offset,
            initial_sets=self.initial_sets,
            safe_sets=self.safe_sets,
            unsafe_sets=self.unsafe_sets,
            show_plot=self.show_plot,
            optimize_with_rle=self.optimize_with_rle,
        )
        self.forward_initialized = True


    def _initialize_backward_reach(self):
        """
        Initializes the backward reach object using the specified quantized reachability class.
        """
        self.backward = self.backward_class(
            n_dims=self.n_dims,
            dt=self.dt,
            quant_params=self.quant_params,
            state_to_cmd=self.state_to_cmd,
            get_dynamics=self.get_dynamics,
            get_offset=self.get_offset,
            initial_sets=self.initial_sets,
            safe_sets=self.safe_sets,
            unsafe_sets=self.unsafe_sets,
            possible_commands=self.possible_commands,
            show_plot=self.show_plot,
            optimize_with_rle=self.optimize_with_rle,
        )
        self.backward_initialized = True


    def set_custom_forward_reach(self, custom_class):
        """
        Sets the forward quantized reachability class to a user-specified implementation.
        """
        self.forward_class = custom_class


    def set_custom_backward_reach(self, custom_class):
        """
        Sets the backward quantized reachability class to a user-specified implementation.
        """
        self.backward_class = custom_class


    def verify_forward(self):
        """
        Performs forward quantized reachability analysis.
        """
        if not self.forward_initialized:
            self._initialize_forward_reach()
        return self._verify(reach_obj=self.forward)


    def verify_backward(self):
        """
        Performs backward quantized reachability analysis.
        """
        if not self.backward_initialized:
            self._initialize_backward_reach()
        return self._verify(reach_obj=self.backward)


    def _verify(self, reach_obj):
        """
        Performs quantized reachability analysis (used for forward and backward).
        With falsification mode enabled, repeatedly refines the quantization
        parameters after an unsafe result in search of a falsifying
        counterexample to the original non-quantized system.

        Parameters:
            reach_obj - a quantized_reach object (forward or backward)

        Returns:
            With falsification mode disabled, returns a Counterexample to the
            quantized system, or None if the quantized system is safe.

            With falsification mode enabled, returns a falsifying Counterexample
            to the original non-quantized system, or None if the quantized
            system is safe.
        """
        while True:
            # Attempt to verify at the given quantization level.
            counterexample = reach_obj.verify_quantized_safety()

            if counterexample is not None:
                # Found a counterexample to the quantized system; quantized system is unsafe.

                ## Coherence check: Simulate the quantized counterexample to
                ## ensure its behavior matches the reachability result.
                sim_is_safe, sim_cmd_seq = self.simulate_point(point=counterexample.init_pt, quantized=True)
                #assert sim_is_safe == False
                #assert sim_cmd_seq == counterexample.cmd_seq, f"Simulated command sequence: {sim_cmd_seq}\nReachability command sequence: {counterexample.cmd_seq}"

            if counterexample is None:
                # Quantized system is safe
                log.info(f"Quantized system is safe under quant_params {reach_obj.quant_params}")
                return None

            elif (counterexample is not None) and (not self.falsify_nonquantized_system):
                # The quantized system is unsafe, and the user does not care
                # about trying to falsify the original (non-quantized) system.
                # Return the counterexample to the quantized system.
                log.info(f"Quantized system is unsafe under quant_params {reach_obj.quant_params}")
                return counterexample

            # Otherwise: the quantized system is unsafe, but the user wants
            # proof that the counterexample is not a "false alarm" (i.e. that it
            # also falsifies the original non-quantized system)
            orig_is_safe, orig_cmd_seq = self.simulate_point(point=counterexample.init_pt, quantized=False)

            if not orig_is_safe:
                # The original (non-quantized) system is also unsafe.
                log.info(f"Original non-quantized system is unsafe with counterexample initial state: {counterexample.init_pt}")
                return counterexample

            # The quantized system is unsafe, but the counterexample does not
            # falsify the original system (i.e. the systems' behaviors do not
            # match). Refine the quantization parameters until the quantized
            # system is safe, or until the real system is falsified.
            log.info(f"Quantized system is unsafe under quant_params {reach_obj.quant_params}, but original non-quantized system was not falsified.")
            log.info(f"Refining quantization parameters.")
            reach_obj.refine_quant_params()
            log.info(f"New quantization parameters: {reach_obj.quant_params}")

            # The user-provided reachability object has been modified during the
            # falsification attempt. It should be re-initialized from scratch
            # before the user initiates a new call to verify.
            self.forward_initialized = False
            self.backward_initialized = False


    def simulate_point(self, *args, **kwargs):
        """
        Simulates a point in the quantized or non-quantized system.

        Returns (is_safe, cmd_seq):
            is_safe - whether or not the simulation avoided the unsafe set
            cmd_seq - the sequence of commands issued by the controller
        """
        if not self.forward_initialized:
            self._initialize_forward_reach()
        return self.forward.simulate_point(*args, **kwargs)

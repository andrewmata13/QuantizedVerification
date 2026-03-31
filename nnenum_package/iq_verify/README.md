# IQ-Verify: A Tool for Verifying Input-Quantized Neural Network Control Systems

IQ-Verify is a tool for verifying input-quantized neural network control systems
(NNCS). Quantizing the inputs to a neural network controller can make the NNCS
easier to analyze, without significantly altering its behavior. IQ-Verify
analyzes the input-quantized NNCS, which can be deployed with a strong safety
guarantee.



## Contents

This section describes the important files included in this repository.

The `examples/` directory contains several applications of the IQ-Verify tool
for analyzing neural network control systems (NNCS).
These examples can be a good starting point for learning how to apply IQ-Verify
to a specific use-case.

The `iq_verify/` directory contains the top-level `iq_verify.py` file, which
defines the `IQVerify` class used in all the examples.
This directory also contains several subdirectories.

The `iq_verify/linprog/` subdirectory contains files related to linear
programming, which allows for efficient computation even in high dimensions.
This technique is used heavily by the set representations in `iq_verify/set_repr/`.
Advanced users may modify the tool to use any LP solver (e.g. Mosek), as long as
they provide a wrapper that conforms to the LP-solver interface defined in
`linprog.py`.
`linprog.py` defines the abstraction for LP-solving that is used
by the `set_repr` interface, and gives several implementations.
`lpinstance.py` and `timerutil.py` contain code used in one of the
implementations.

The `iq_verify/quant_reach/` subdirectory contains files related to reachability
analysis for quantized NNCS.
The `quantized_reach.py` file defines the interface for a quantized reachability
analysis object, and provides default implementations for many methods
(e.g. quantizing along uniform axis-aligned quantization boundaries).
`quantized_forward_reach.py` and `quantized_backward_reach.py` are siblings
that both inherit from the `quantized_reach.py` interface.
Generally, users interested in defining their own quantized reachability objects
should extend one of these siblings, instead of extending the interface directly.
See the relevant section below for more information on defining a custom
quantized reachability class.

The `iq_verify/set_repr/` subdirectory contains files related to describing and
transforming statesets (i.e., set representations).
The `set_repr.py` file describes the interface that `IQVerify` expects from a
set representation.
Several common set representations are provided, such as `h_polytope.py` and `star_set.py`.
Users may create additional set representations, as long as they adhere to the
interface in `set_repr.py`.

The `rep/` directory contains scripts used to reproduce figures and tables from
the paper (under submission).




## Tool Usage

### Basic: Quantized Forward/Backward Reachability

In some cases, the assumptions made by the default implementations of the
quantized reachability objects are satisfactory for an application.
Some of these assumptions include:

- The state space should be quantized along axis-aligned quantization boundaries.
- A state is unsafe if and only if it reaches a pre-defined unsafe subset of the
state space.
- An execution terminates safely if and only if it reaches a pre-defined safe
subset of the state space (e.g., invariant set).
- The control command can be determined directly from the N-dimensional state,
with no other info needed.
- The affine dynamics can be determined directly from the control command,
with no other info needed.

When a system does not require custom behavior that contradicts these
assumptions, IQ-Verify can be constructed and run very easily.
The harmonic oscillator examples in `examples/harmonic_oscillator/` demonstrate
a simple case where IQ-Verify is provided with the initial, safe, and unsafe
sets; the affine dynamics; the quantization parameters; and other required info
such as timestep size. In such cases, verification can be performed directly,
without the need to implement any custom quantized reachability objects.




### Advanced: Custom Quantized Reachability

Users may wish to override or extend the default behaviors of IQ-Verify.
Some potential reasons for extending the code are as follows:

- Custom quantization techniques, such as non-uniform or non-axis-aligned quantization.
- Unsafe (or safe) termination conditions based on custom set properties, such
as failure due to timeout.
- Control command based on information besides N-dimensional state, such as
checking command history in order to inject bias toward repeating the previous command.

Typically, the user should create a custom forward or backward reachability
object that inherits from `QuantizedForwardReach` or `QuantizedBackwardReach`,
respectively.
An example can be seen in `examples/robot_navigation/robot.py`.
In this example, the `FwdRobotReach` class modifies the `find_counterexample`
method to check the elapsed time property of each reachable set.
The custom method flags as unsafe any reachable set violating the timeout
condition.
The tool is notified to use the custom forward reachability object via the
`iqv.set_custom_forward_reach(FwdRobotReach)` method call.
A similar method exists for backward reachability as well.

Note: Simulation is a functionality of the `QuantizedForwardReach` class.
Consequently, use-cases where a custom backward reachability object is used must
also be capable of constructing a corresponding forward reachability object, if
the ability to simulate a point is desired.
The ACAS Xu example `examples/acasxu/backreach_setup.py` includes a custom
backward reachability class that overrides many default behaviors, as well as
a custom forward reachability class meant only to provide simulation capabilities.
(The ACAS Xu example does not perform forward reachability, but does require
forward simulation. Therefore, both custom classes are needed.)




## Docker Setup Instructions

The Docker image can be viewed on Docker Hub, at the following web address:
https://hub.docker.com/r/wehbedoug/iqverify

In the host machine terminal, pull the Docker image from Docker Hub using
the following command:

```bash
sudo docker pull wehbedoug/iqverify:20241226
```

Run the Docker image in interactive mode.
Note that the command below will create a directory `./images/` on the host system,
which will be mounted in the Docker container as a volume.

```bash
sudo docker run -it --rm \
    -v ./images:/home/images \
    wehbedoug/iqverify:20241226
```

This should result in launching a `bash` shell inside the `/home/` directory of the container, running as the `root` user.
The `/home/images/` directory inside the container is a one-to-one mapping of the `./images/` directory on the host machine.

### (Optional) Host Machine Permissions Workaround

If for some reason mounting a host directory in the container with `-v` is causing permissions issues, there is a workaround.
If permissions issues occur related to the mounted directory, launch the container without the `-v` flag:

```bash
sudo docker run -it --rm \
    wehbedoug/iqverify:20241226
```

Images will still be created by the scripts and placed in the `/home/images/` directory of the container.
The images can be manually downloaded from a running container to the host machine.
Transferring the images from a running container requires knowledge of the
running container's `<container-id>`. This can be obtained via the following
command, in a second host machine terminal. Look for the `CONTAINER ID` field in
the output:

```bash
sudo docker container ps
```

An example of manually downloading a figure is shown below, for `figure3.png`:

```bash
sudo docker cp \
    <container-id>:/home/images/figure3.png \
    ./figure3_host_copy.png
```




## Examples

### Harmonic Oscillator

This example uses simple dynamics without a neural network.
It ignores many features related to quantization and NNCS, and thus primarily
serves to ensure the environment is configured correctly.

```bash
python ./examples/harmonic_oscillator/safe.py

python ./examples/harmonic_oscillator/unsafe.py
```




### Inverted Pendulum

This example is based on the ARCH AINNCS competition.
An example invocation is given below:

```bash
python examples/inverted_pendulum/single.py --q_theta=0.1 --q_nu=0.1 --q_tau=0.1 \
    --forward=True --simulate=True --show_plot=True --savefig=True
```




### Robot Navigation

This example is based on the ARCH AINNCS competition.
An example invocation is given below:

```bash
python examples/robot_navigation/robot.py --q_x=0.1 --q_y=0.1 --q_nu=0.1 --q_theta=0.1 \
    --q_u1=0.1 --q_u2=0.1 --forward=True --simulate=True \
    --falsify=False --use_robust_nn=False --show_plot=True --savefig=True
```





### ACAS Xu

This example performs quantized verification for the neural network compression
of the ACAS Xu collision avoidance system.
Unlike many of the other examples, the state space in ACAS Xu is not quantized
by splitting along an axis-aligned grid.
Instead, quantization is performed as described in
"Neural Network Compression of ACAS Xu Prototype is Unsafe: Closed-Loop
Verification through Quantized State Backreachability" by Stanley Bak and
Hoang-Dung Tran, available at https://arxiv.org/abs/2201.06626.

View the command-line arguments:

```bash
python ./examples/acasxu/main.py --help
```

The `--collision` argument should be supplied as a string in the following format:

```python
--collision="(dx, dy, theta_own, v_own, v_int, a_prev)"
```

Below are a few interesting examples:

- Safe: `--collision="(-1, -1, 17, 5, 8, 4)"`
    - In only a handful of steps, IQ-Verify proves that there is no feasible
      path from the collision to an initial state.

- Safe: `--collision="(-1, 0, 46, 1, 11, 3)"`
    - Pops ~13,300 sets from the queue before all paths are deemed infeasible.

- Unsafe: `--collision="(0, 0, 191, 1, 5, 4)"`
    - Pops ~700 sets before finding an unsafe initial state.
    - Note: This only serves as a counterexample to the _quantized_ system.
      More work would be required to falsify the original non-quantized system.

- Unsafe: `--collision="(-1, -1, 182, 2, 6, 4)"`
    - Pops ~52,100 sets before finding an unsafe initial state.
    - Note: Same as above, only proves that the _quantized_ system is unsafe.

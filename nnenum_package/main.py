import numpy as np
import matplotlib.pyplot as plt
import swiglpk as glpk

import nnenum
from nnenum.nnenum import set_control_settings, set_exact_settings, make_spec
from nnenum.settings import Settings
from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.result import Result
from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum.specification import Specification, DisjunctiveSpec
from nnenum.vnnlib import get_num_inputs_outputs, read_vnnlib_simple
from nnenum.lpinstance import SwigArray

from iq_verify.set_repr import star_set
from iq_verify.quant_reach.quantization_utils import QuantizationUtils

# Example
vnnlib_filename="specification.vnnlib"
onnx_filename="simple.onnx"

spec_list, input_dtype = make_spec(
    vnnlib_filename=vnnlib_filename,
    onnx_filename=onnx_filename,
)

network = load_onnx_network_optimized(onnx_filename)

# Options are "control", "image", "exact"
set_exact_settings()

# Set RESULT_SAVE_STARS to True in nnenum settings
Settings.RESULT_SAVE_STARS = True
#Settings.TRY_QUICK_OVERAPPROX = True   # default is False for set_exact_settings
Settings.OVERAPPROX_BOTH_BOUNDS = True  # default is False for set_exact_settings
Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX  # default is Settings.BRANCH_EXACT for set_exact_settings

count = 1
nnenum_stars = []
for init_box, spec in spec_list:
    init_box = np.array(init_box, dtype=input_dtype)

    res = enumerate_network(init_box, network, spec)
    result_str = res.result_str
    nnenum_stars += res.stars



################################################################################
# Plot the stars, as well as the quantized cell centers
# nnenum lp_star -> IQ-Verify StarSet
################################################################################

for nnenum_star in nnenum_stars:
    iqv_star = star_set.StarSet.from_nnenum_LpStar(nnenum_star)
    iqv_star.plot("red", fill=True, alpha=0.1)

    quant_params = [2.5, 2.0]
    pts = QuantizationUtils.stateset_to_qpoints(iqv_star, quant_params)
    for pt in pts:
        plt.plot(pt[0], pt[1], marker="*", markersize=6, color="blue")

plt.grid()
plt.show()
plt.close()

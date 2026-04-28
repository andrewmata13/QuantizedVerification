"""Subprocess wrapper: runs nnenum with control settings and writes result as JSON."""
import os, sys, time, json

os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nnenum_package", "nnenum", "src"))

from nnenum.settings import Settings
from nnenum.nnenum import set_control_settings
from nnenum.enumerate import enumerate_network
from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.specification import Specification, DisjunctiveSpec
from nnenum.vnnlib import read_vnnlib_simple, get_num_inputs_outputs

Settings.CHECK_SINGLE_THREAD_BLAS = False
set_control_settings()
Settings.TIMEOUT = int(sys.argv[3])
Settings.RESULT_SAVE_STARS = False
Settings.PRINT_OUTPUT = True
Settings.PRINT_PROGRESS = True
Settings.NUM_PROCESSES = 32

onnx_path = sys.argv[1]
spec_path = sys.argv[2]
out_json = sys.argv[4]

n_in, n_out, inp_dtype = get_num_inputs_outputs(onnx_path)
network = load_onnx_network_optimized(onnx_path)

vnnlib_data = read_vnnlib_simple(spec_path, n_in, n_out)
box, action_spec_list = vnnlib_data[0]
init_box = np.array(box, dtype=inp_dtype)

if len(action_spec_list) == 1:
    mat, rhs = action_spec_list[0]
    spec = Specification(mat, rhs)
else:
    spec = DisjunctiveSpec([Specification(m, r) for m, r in action_spec_list])

t0 = time.time()
res = enumerate_network(init_box, network, spec)
elapsed = time.time() - t0

raw = res.result_str
if raw == "safe":
    label = "safe"
elif raw.startswith("unsafe"):
    label = "unsafe"
elif raw == "timeout":
    label = "timeout"
else:
    label = raw

result = {
    "result": label,
    "time": round(elapsed, 2),
    "raw_result_str": raw,
}

with open(out_json, "w") as f:
    json.dump(result, f)

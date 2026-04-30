"""
run_nnenum_full.py

Run nnenum (BRANCH_EGO, sound) on all 12 specs for all 4 full networks:
  - HalfCheetah baseline [512,512]
  - HalfCheetah latent3  (encoder + controller, full_network.onnx)
  - Hopper    baseline [512,512]
  - Hopper    latent3  (encoder + controller, full_network.onnx)

Timeout: 1800s per spec (same as CROWN comparison).
Results saved incrementally to /tmp/nnenum_full_results.json.
"""
import os, sys, time, json
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["MUJOCO_GL"] = "egl"

import numpy as np

from nnenum.settings import Settings
from nnenum.enumerate import enumerate_network
from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.specification import Specification, DisjunctiveSpec
from nnenum.vnnlib import read_vnnlib_simple, get_num_inputs_outputs

TIMEOUT = 1800
OUTFILE = "/tmp/nnenum_full_results.json"

NETWORKS = [
    ("HalfCheetah-v4", "baseline",
     "sac_sweep_runs/HalfCheetah-v4/baseline/seed0/full_network.onnx"),
    ("HalfCheetah-v4", "latent3",
     "sac_sweep_runs/HalfCheetah-v4/latent3/seed0/full_network.onnx"),
    ("Hopper-v5", "baseline",
     "sac_sweep_runs/Hopper-v5/baseline512/seed0/full_network.onnx"),
    ("Hopper-v5", "latent3",
     "sac_sweep_runs/Hopper-v5/latent3/seed0/full_network.onnx"),
]


def nnenum_settings():
    Settings.CHECK_SINGLE_THREAD_BLAS = False
    Settings.BRANCH_MODE    = Settings.BRANCH_EGO
    Settings.TRY_QUICK_OVERAPPROX   = True
    Settings.OVERAPPROX_BOTH_BOUNDS = True
    Settings.TIMEOUT        = TIMEOUT
    Settings.RESULT_SAVE_STARS = False
    Settings.PRINT_OUTPUT   = False
    Settings.PRINT_PROGRESS = False
    Settings.NUM_PROCESSES  = 32


def run_one(onnx_path, spec_path):
    nnenum_settings()
    n_in, n_out, inp_dtype = get_num_inputs_outputs(onnx_path)
    network  = load_onnx_network_optimized(onnx_path)

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

    raw = res.result_str   # "safe", "unsafe", "unsafe (unconfirmed)", "timeout", "none"
    if raw == "safe":
        label = "safe"
    elif raw.startswith("unsafe"):
        label = "unsafe"
    elif raw == "timeout":
        label = "timeout"
    else:
        label = raw

    return label, round(elapsed, 2)


results = {}
try:
    with open(OUTFILE) as f:
        results = json.load(f)
except FileNotFoundError:
    pass

for env, net_label, onnx_path in NETWORKS:
    key = f"{env} / {net_label}"
    if key not in results:
        results[key] = {}

    spec_dir = f"specs/{env}"
    print(f"\n{'='*60}")
    print(f"  {key}")
    print(f"{'='*60}")

    for sid in range(1, 13):
        spec_name = f"spec_{sid}"
        if spec_name in results[key]:
            r = results[key][spec_name]
            print(f"  {spec_name}: {r['result']}  {r['time']}s  [cached]")
            continue

        spec_path = os.path.join(spec_dir, f"{spec_name}.vnnlib")
        if not os.path.exists(spec_path):
            print(f"  {spec_name}: MISSING — skip")
            continue

        print(f"  {spec_name} ... ", end="", flush=True)
        result, elapsed = run_one(onnx_path, spec_path)
        results[key][spec_name] = {"result": result, "time": elapsed}
        print(f"{result}  {elapsed}s")

        with open(OUTFILE, "w") as f:
            json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {OUTFILE}")
print(json.dumps(results, indent=2))

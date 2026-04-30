"""
run_nnenum_comparison.py

Run nnenum directly on full_network.onnx for all specs using a fresh subprocess
per spec (avoids multiprocessing state contamination between calls).

Usage:
    python run_nnenum_comparison.py
    python run_nnenum_comparison.py --timeout 600 --out /tmp/nnenum_results.json
"""

import os, sys, json, time, argparse, subprocess, tempfile

NETWORKS = [
    ("HalfCheetah baseline [512,512]",
     "sac_sweep_runs/HalfCheetah-v4/baseline/seed0/full_network.onnx",
     [f"specs/HalfCheetah-v4/spec_{i}.vnnlib" for i in range(1, 9)]),
    ("HalfCheetah latent3",
     "sac_sweep_runs/HalfCheetah-v4/latent3/seed0/full_network.onnx",
     [f"specs/HalfCheetah-v4/spec_{i}.vnnlib" for i in range(1, 9)]),
    ("Hopper baseline [512,512]",
     "sac_sweep_runs/Hopper-v5/baseline512/seed0/full_network.onnx",
     [f"specs/Hopper-v5/spec_{i}.vnnlib" for i in range(1, 5)]),
    ("Hopper latent3",
     "sac_sweep_runs/Hopper-v5/latent3/seed0/full_network.onnx",
     [f"specs/Hopper-v5/spec_{i}.vnnlib" for i in range(1, 5)]),
]

WORKER = """
import os, sys, json, time, numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "{workdir}")

from nnenum.nnenum import set_exact_settings
from nnenum.settings import Settings
from nnenum.enumerate import enumerate_network
from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum.specification import Specification
from nnenum.vnnlib import read_vnnlib_simple, get_num_inputs_outputs

set_exact_settings()
Settings.RESULT_SAVE_STARS = False
Settings.OVERAPPROX_BOTH_BOUNDS = True
Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
Settings.TIMEOUT = {timeout}

onnx_path = {onnx!r}
spec_path  = {spec!r}

try:
    net = load_onnx_network_optimized(onnx_path)
except Exception:
    net = load_onnx_network(onnx_path)

n_in, n_out, _ = get_num_inputs_outputs(onnx_path)
vnnlib_data = read_vnnlib_simple(spec_path, n_in, n_out)
box, constraints = vnnlib_data[0]
init_box = np.array(box, dtype=np.float32)

if constraints:
    mat, rhs = constraints[0]
    spec = Specification(np.array(mat, dtype=np.float32),
                         np.array(rhs, dtype=np.float32))
else:
    spec = Specification(np.eye(n_out, dtype=np.float32),
                         np.full(n_out, -1e9, dtype=np.float32))

t0 = time.time()
result = enumerate_network(init_box, net, spec)
elapsed = time.time() - t0

status = str(result.result_str).lower()
if "unsafe" in status or "violated" in status:
    label = "unsafe"
elif "safe" in status and "incomplete" not in status:
    label = "safe"
elif "timeout" in status:
    label = "timeout"
else:
    label = status

print(json.dumps({{"result": label, "time": round(elapsed, 2)}}))
"""


def run_one(onnx_path, spec_path, timeout, workdir):
    code = WORKER.format(
        workdir=workdir,
        timeout=timeout,
        onnx=onnx_path,
        spec=spec_path,
    )
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MUJOCO_GL"] = "egl"

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            timeout=timeout + 30,
            env=env,
        )
        elapsed = time.time() - t0
        # Find JSON in output
        for line in reversed(proc.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        # No JSON found — check stderr for clues
        err = proc.stderr[-200:] if proc.stderr else ""
        return {"result": f"error: {err.strip()}", "time": round(elapsed, 2)}
    except subprocess.TimeoutExpired:
        return {"result": "timeout", "time": round(time.time() - t0, 2)}
    except Exception as e:
        return {"result": f"error: {e}", "time": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="/tmp/nnenum_results.json")
    args = ap.parse_args()

    workdir = os.path.dirname(os.path.abspath(__file__))
    results = {}

    for net_label, onnx_path, spec_paths in NETWORKS:
        if not os.path.exists(onnx_path):
            print(f"SKIP {net_label} — {onnx_path} missing")
            continue

        print(f"\n=== {net_label} ===")
        results[net_label] = {}

        for spec_path in spec_paths:
            if not os.path.exists(spec_path):
                continue
            spec_name = os.path.basename(spec_path).replace(".vnnlib", "")
            print(f"  {spec_name} ... ", end="", flush=True)

            r = run_one(onnx_path, spec_path, args.timeout, workdir)
            results[net_label][spec_name] = r
            print(f"{r['result']}  {r['time']}s", flush=True)

            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nDone. Results in {args.out}")


if __name__ == "__main__":
    main()

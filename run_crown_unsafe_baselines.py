"""Run CROWN on specs 9-12 (unsafe) for HalfCheetah and Hopper baselines."""
import os, json, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_abcrown import run_abcrown

RUNS = [
    ("HalfCheetah baseline",
     "sac_sweep_runs/HalfCheetah-v4/baseline/seed0/full_network.onnx",
     [f"specs/HalfCheetah-v4/spec_{i}.vnnlib" for i in range(9, 13)]),
    ("Hopper baseline",
     "sac_sweep_runs/Hopper-v5/baseline512/seed0/full_network.onnx",
     [f"specs/Hopper-v5/spec_{i}.vnnlib" for i in range(9, 13)]),
]
OUTFILE = "/tmp/crown_unsafe_baselines.json"
TIMEOUT = 1800

results = {}
for label, onnx, specs in RUNS:
    results[label] = {}
    for spec_path in specs:
        name = os.path.basename(spec_path).replace(".vnnlib", "")
        print(f"{label} / {name} ...", flush=True)
        result, elapsed = run_abcrown(onnx, spec_path, TIMEOUT, batch_size=256)
        results[label][name] = {"result": result, "time": round(elapsed, 2)}
        print(f"  {result}  {elapsed:.1f}s", flush=True)
        with open(OUTFILE, "w") as f:
            json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))

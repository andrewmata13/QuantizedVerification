"""Re-run specs 2 and 3 for Hopper [512,512] baseline (previous results were spurious)."""
import os, json, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_abcrown import run_abcrown

ONNX = "sac_sweep_runs/Hopper-v5/baseline512/seed0/full_network.onnx"
SPECS = [
    ("spec_2", "specs/Hopper-v5/spec_2.vnnlib"),
    ("spec_3", "specs/Hopper-v5/spec_3.vnnlib"),
]
OUTFILE = "/tmp/crown_hopper_baseline_recheck.json"
TIMEOUT = 1800

results = {}
for name, spec_path in SPECS:
    print(f"Running {name} ...", flush=True)
    result, elapsed = run_abcrown(ONNX, spec_path, TIMEOUT, batch_size=256)
    results[name] = {"result": result, "time": round(elapsed, 2)}
    print(f"  {name}: {result}  {elapsed:.1f}s", flush=True)
    with open(OUTFILE, "w") as f:
        json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))

"""Run CROWN on Hopper unsafe specs one at a time, writing results to JSON immediately."""
import os, json, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["MUJOCO_GL"] = "egl"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_abcrown import run_abcrown

ONNX = "sac_sweep_runs/Hopper-v5/latent3/seed0/full_network.onnx"
SPECS = [
    ("unsafe_1", "specs/Hopper-v5/unsafe_spec_1_latent3.vnnlib"),
    ("unsafe_2", "specs/Hopper-v5/unsafe_spec_2_latent3.vnnlib"),
    ("unsafe_3", "specs/Hopper-v5/unsafe_spec_3_latent3.vnnlib"),
    ("unsafe_4", "specs/Hopper-v5/unsafe_spec_4_latent3.vnnlib"),
]
OUTFILE = "/tmp/crown_hopper_unsafe_results.json"
TIMEOUT = 1800

results = {}
for name, spec_path in SPECS:
    print(f"Running {name} ...", flush=True)
    result, elapsed = run_abcrown(ONNX, spec_path, TIMEOUT, batch_size=256)
    results[name] = {"result": result, "time": round(elapsed, 2)}
    print(f"  {name}: {result}  {elapsed:.1f}s", flush=True)
    with open(OUTFILE, "w") as f:
        json.dump(results, f, indent=2)

print(f"\nAll done. Results in {OUTFILE}")
print(json.dumps(results, indent=2))

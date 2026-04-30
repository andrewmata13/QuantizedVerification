"""
Run alpha-beta CROWN on latent3 specs 9-12 for HalfCheetah and Hopper.
Saves results to /tmp/crown_unsafe_latent3.json.
"""
import os, sys, time, json
os.environ["MUJOCO_GL"] = "egl"

CROWN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "alpha-beta-CROWN", "complete_verifier")
sys.path.insert(0, CROWN_DIR)

import torch

TIMEOUT = 1800
OUTFILE = "/tmp/crown_unsafe_latent3.json"

JOBS = [
    ("HalfCheetah-v4", "latent3",
     "sac_sweep_runs/HalfCheetah-v4/latent3/seed0/full_network.onnx"),
    ("Hopper-v5", "latent3",
     "sac_sweep_runs/Hopper-v5/latent3/seed0/full_network.onnx"),
]

_SAFE_STATUSES = {"verified", "safe", "safe-incomplete"}
_UNSAFE_STATUSES = {"unsafe-pgd", "unsafe-bab", "falsified"}


def run_one(onnx_path, vnnlib_path, timeout):
    from api import ABCrownSolver, VerificationSpec, ConfigBuilder
    from api import _deep_update, _clone_config, _ensure_config_defaults
    import arguments

    device = "cuda" if torch.cuda.is_available() else "cpu"
    builder = (
        ConfigBuilder.from_defaults()
        .set(general__device=device)
        .set(solver__batch_size=1024)
        .set(bab__timeout=timeout)
    )
    cfg = builder()

    _ensure_config_defaults()
    new_cfg = _clone_config(arguments.Config.all_args)
    _deep_update(new_cfg, cfg)
    arguments.Config.all_args = new_cfg
    arguments.Config.update_arguments()

    spec = VerificationSpec.build_spec(vnnlib_path=os.path.abspath(vnnlib_path))
    solver = ABCrownSolver(spec, os.path.abspath(onnx_path), config=cfg)
    t0 = time.time()
    result = solver.solve()
    elapsed = time.time() - t0

    status = str(getattr(result, "status", "unknown"))
    if status in _SAFE_STATUSES:
        label = "safe"
    elif status in _UNSAFE_STATUSES:
        label = "unsafe"
    elif "timeout" in status.lower() or "unknown" in status.lower():
        label = "timeout"
    else:
        label = status

    return label, round(elapsed, 2)


results = {}
try:
    with open(OUTFILE) as f:
        results = json.load(f)
except FileNotFoundError:
    pass

for env, net_label, onnx_path in JOBS:
    key = f"{env} / {net_label}"
    if key not in results:
        results[key] = {}

    print(f"\n{'='*60}")
    print(f"  {key}")
    print(f"{'='*60}")

    for sid in range(9, 13):
        spec_name = f"spec_{sid}"
        if spec_name in results[key]:
            r = results[key][spec_name]
            print(f"  {spec_name}: {r['result']}  {r['time']}s  [cached]")
            continue

        spec_path = f"specs/{env}/{spec_name}.vnnlib"
        if not os.path.exists(spec_path):
            print(f"  {spec_name}: MISSING")
            continue

        print(f"  {spec_name} ... ", end="", flush=True)
        result, elapsed = run_one(onnx_path, spec_path, TIMEOUT)
        results[key][spec_name] = {"result": result, "time": elapsed}
        print(f"{result}  {elapsed}s")

        with open(OUTFILE, "w") as f:
            json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {OUTFILE}")
print(json.dumps(results, indent=2))

"""Subprocess wrapper: runs alpha-beta CROWN and writes result as JSON."""
import os, sys, time, json

os.environ["MUJOCO_GL"] = "egl"

import torch

CROWN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "alpha-beta-CROWN", "complete_verifier")
sys.path.insert(0, CROWN_DIR)

from api import ABCrownSolver, VerificationSpec, ConfigBuilder
from api import _deep_update, _clone_config, _ensure_config_defaults
import arguments

onnx_path = os.path.abspath(sys.argv[1])
spec_path = os.path.abspath(sys.argv[2])
timeout = int(sys.argv[3])
out_json = sys.argv[4]

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 256

builder = (
    ConfigBuilder.from_defaults()
    .set(general__device=device)
    .set(solver__batch_size=batch_size)
    .set(bab__timeout=timeout)
)
cfg = builder()

_ensure_config_defaults()
new_cfg = _clone_config(arguments.Config.all_args)
_deep_update(new_cfg, cfg)
arguments.Config.all_args = new_cfg
arguments.Config.update_arguments()

_SAFE = {"verified", "safe", "safe-incomplete"}
_UNSAFE = {"unsafe-pgd", "unsafe-bab", "falsified"}

spec = VerificationSpec.build_spec(vnnlib_path=spec_path)
solver = ABCrownSolver(spec, onnx_path, config=cfg)
t0 = time.time()
result = solver.solve()
elapsed = time.time() - t0

status = str(getattr(result, "status", "unknown"))
if status in _SAFE:
    label = "safe"
elif status in _UNSAFE:
    label = "unsafe"
elif "timeout" in status.lower() or "unknown" in status.lower():
    label = "timeout"
else:
    label = status

out = {
    "result": label,
    "time": round(elapsed, 2),
    "raw_status": status,
    "device": device,
    "batch_size": batch_size,
}

with open(out_json, "w") as f:
    json.dump(out, f)

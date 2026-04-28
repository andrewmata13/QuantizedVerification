"""Subprocess wrapper: runs verify() and writes result as JSON to a temp file."""
import os, sys, json

os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nnenum_package", "nnenum", "src"))

from nnenum.settings import Settings
Settings.CHECK_SINGLE_THREAD_BLAS = False
from verify_policy import verify
from nnenum.vnnlib import get_num_inputs_outputs

enc_onnx = sys.argv[1]
spec_path = sys.argv[2]
ctrl_pth = sys.argv[3]
quant_step = float(sys.argv[4])
out_json = sys.argv[5]

n_in, _, _ = get_num_inputs_outputs(enc_onnx)
full_onnx = sys.argv[6]
_, n_act, _ = get_num_inputs_outputs(full_onnx)

r = verify(
    enc_onnx, spec_path, ctrl_pth,
    n_inputs=n_in, n_actions=n_act,
    quant_step=quant_step,
    overapprox=True, complete=True,
)

result = {
    "result": r["result"],
    "time": round(r["t_total"], 4),
    "t_encoder": round(r["t_encoder"], 4),
    "t_grid": round(r["t_grid"], 4),
    "t_ctrl": round(r["t_ctrl"], 4),
    "t_check": round(r["t_check"], 4),
    "n_stars": r["n_stars"],
    "n_cells": r["n_cells"],
    "n_violations": len(r["violations"]),
}

with open(out_json, "w") as f:
    json.dump(result, f)

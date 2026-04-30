"""Subprocess wrapper: runs verify() with a QAT INT8 ORT controller."""
import os, sys, json

os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nnenum_package", "nnenum", "src"))

from nnenum.settings import Settings
Settings.CHECK_SINGLE_THREAD_BLAS = False
from verify_policy import verify
from nnenum.vnnlib import get_num_inputs_outputs


class OrtController:
    def __init__(self, onnx_path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            onnx_path, opts, providers=["CPUExecutionProvider"])
        self.inp_name = self.sess.get_inputs()[0].name

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        out = self.sess.run(None, {self.inp_name: x})[0]
        return torch.from_numpy(out)

    def to(self, device):
        return self

    def eval(self):
        return self

    def modules(self):
        return []


enc_onnx = sys.argv[1]
spec_path = sys.argv[2]
int8_onnx = sys.argv[3]
quant_step = float(sys.argv[4])
out_json = sys.argv[5]
full_onnx = sys.argv[6]

n_in, _, _ = get_num_inputs_outputs(enc_onnx)
_, n_act, _ = get_num_inputs_outputs(full_onnx)

ctrl = OrtController(int8_onnx)

r = verify(
    enc_onnx, spec_path, ctrl,
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

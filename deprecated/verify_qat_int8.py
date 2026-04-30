"""
Run verification on QAT INT8 controller (ORT static) vs float32 for HC specs 1-10.
"""
import os, time, numpy as np, torch
os.environ["MUJOCO_GL"] = "egl"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from nnenum.settings import Settings
Settings.CHECK_SINGLE_THREAD_BLAS = False
from nnenum.vnnlib import get_num_inputs_outputs
from verify_policy import verify

import onnxruntime as ort


class OrtController:
    """Wraps an ORT session to look like a PyTorch module for verify_policy."""
    def __init__(self, onnx_path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0
        opts.inter_op_num_threads = 0
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


RUN_DIR = "sac_sweep_runs/HalfCheetah-v4/latent3/seed0"
QUANT_STEP = 0.05

enc_onnx = os.path.join(RUN_DIR, "encoder.onnx")
full_onnx = os.path.join(RUN_DIR, "full_network.onnx")
ctrl_f32_path = os.path.join(RUN_DIR, "latent_controller_full.pth")
ctrl_qat_i8_onnx = os.path.join(RUN_DIR, "latent_controller_qat_static_int8.onnx")

n_in, _, _ = get_num_inputs_outputs(enc_onnx)
_, n_act, _ = get_num_inputs_outputs(full_onnx)

ort_ctrl = OrtController(ctrl_qat_i8_onnx)

print(f"{'Spec':<10} {'QAT INT8 result':<16} {'time':>10}")
print(f"{'─'*10} {'─'*16} {'─'*10}")

for sid in range(1, 11):
    spec_path = f"specs/HalfCheetah-v4/spec_{sid}.vnnlib"

    r_i8 = verify(enc_onnx, spec_path, ort_ctrl,
                  n_inputs=n_in, n_actions=n_act,
                  quant_step=QUANT_STEP, complete=True)

    print(f"spec_{sid:<5} {r_i8['result']:<16} {r_i8['t_total']:>9.2f}s")

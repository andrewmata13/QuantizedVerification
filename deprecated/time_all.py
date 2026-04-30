"""Rerun timing experiments for HC and Hopper, f32 and QAT INT8, all on CPU."""
import os, sys, time
os.environ["MUJOCO_GL"] = "egl"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np

from nnenum.settings import Settings
Settings.CHECK_SINGLE_THREAD_BLAS = False
from nnenum.vnnlib import get_num_inputs_outputs
from verify_policy import verify

import onnxruntime as ort


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


ENVS = [
    {
        "name": "HalfCheetah-v4",
        "run_dir": "sac_sweep_runs/HalfCheetah-v4/latent3/seed0",
        "specs": list(range(1, 11)),
        "quant_step": 0.05,
    },
    {
        "name": "Hopper-v5",
        "run_dir": "sac_sweep_runs/Hopper-v5/latent3/seed0",
        "specs": list(range(1, 11)),
        "quant_step": 0.05,
    },
]

for env in ENVS:
    rd = env["run_dir"]
    enc_onnx = os.path.join(rd, "encoder.onnx")
    full_onnx = os.path.join(rd, "full_network.onnx")
    ctrl_f32_path = os.path.join(rd, "latent_controller_full.pth")
    ctrl_i8_onnx = os.path.join(rd, "latent_controller_qat_static_int8.onnx")

    n_in, _, _ = get_num_inputs_outputs(enc_onnx)
    _, n_act, _ = get_num_inputs_outputs(full_onnx)

    has_i8 = os.path.exists(ctrl_i8_onnx)

    print(f"\n{'='*90}")
    print(f"  {env['name']}  (quant_step={env['quant_step']}, CPU only)")
    print(f"{'='*90}")

    hdr = (f"{'Spec':<10} {'Result':<8} {'Stars':>6} {'Cells':>8} "
           f"{'nnenum':>8} {'grid':>8} {'ctrl':>8} {'check':>8} {'total':>8}")
    sep = (f"{'─'*10} {'─'*8} {'─'*6} {'─'*8} "
           f"{'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    # ── Float32 ──
    print(f"\n  Float32 controller (CPU):")
    print(f"  {hdr}")
    print(f"  {sep}")

    for sid in env["specs"]:
        spec_path = f"specs/{env['name']}/spec_{sid}.vnnlib"
        if not os.path.exists(spec_path):
            print(f"  spec_{sid:<5} MISSING")
            continue
        r = verify(enc_onnx, spec_path, ctrl_f32_path,
                   n_inputs=n_in, n_actions=n_act,
                   quant_step=env["quant_step"], complete=True)
        pct_enc = r['t_encoder'] / r['t_total'] * 100
        pct_grid = r['t_grid'] / r['t_total'] * 100
        pct_ctrl = r['t_ctrl'] / r['t_total'] * 100
        pct_chk = r['t_check'] / r['t_total'] * 100
        print(f"  spec_{sid:<5} {r['result']:<8} {r['n_stars']:>6} {r['n_cells']:>8,} "
              f"{r['t_encoder']:>7.3f}s {r['t_grid']:>7.3f}s {r['t_ctrl']:>7.3f}s "
              f"{r['t_check']:>7.3f}s {r['t_total']:>7.2f}s")

    # ── QAT INT8 ──
    if has_i8:
        ort_ctrl = OrtController(ctrl_i8_onnx)
        print(f"\n  QAT INT8 controller (ORT CPU, 1 thread):")
        print(f"  {hdr}")
        print(f"  {sep}")

        for sid in env["specs"]:
            spec_path = f"specs/{env['name']}/spec_{sid}.vnnlib"
            if not os.path.exists(spec_path):
                print(f"  spec_{sid:<5} MISSING")
                continue
            r = verify(enc_onnx, spec_path, ort_ctrl,
                       n_inputs=n_in, n_actions=n_act,
                       quant_step=env["quant_step"], complete=True)
            print(f"  spec_{sid:<5} {r['result']:<8} {r['n_stars']:>6} {r['n_cells']:>8,} "
                  f"{r['t_encoder']:>7.3f}s {r['t_grid']:>7.3f}s {r['t_ctrl']:>7.3f}s "
                  f"{r['t_check']:>7.3f}s {r['t_total']:>7.2f}s")
    else:
        print(f"\n  No QAT INT8 model found at {ctrl_i8_onnx}")

print("\nDone.")

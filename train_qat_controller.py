"""
train_qat_controller.py

QAT distillation for latent controller → ORT static INT8.

The f32 controller has extreme activation outliers (absmax 6800 vs mean 53),
so per-tensor INT8 activation quantization destroys most neurons. QAT training
with fake quantization forces the network to adapt its weight distribution
so activations stay in a quantization-friendly range.

Usage:
    python train_qat_controller.py --env HalfCheetah-v4
    python train_qat_controller.py --env HalfCheetah-v4 --epochs 100 --lr 3e-4
"""
import os, argparse, copy
os.environ["MUJOCO_GL"] = "egl"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, quant_pre_process

CONFIGS = {
    "HalfCheetah-v4": {
        "run_dir": "sac_sweep_runs/HalfCheetah-v4/latent3/seed0",
        "quant_step": 0.05,
    },
    "Hopper-v5": {
        "run_dir": "sac_sweep_runs/Hopper-v5/latent3/seed0",
        "quant_step": 0.1,
    },
}


def collect_data(run_dir, env_name, n_samples=200_000, quant_step=0.05):
    enc = torch.load(os.path.join(run_dir, "encoder_full.pth"), weights_only=False).cpu().eval()
    ctrl = torch.load(os.path.join(run_dir, "latent_controller_full.pth"), weights_only=False).cpu().eval()

    env = DummyVecEnv([lambda: gym.make(env_name)])
    vn = VecNormalize.load(os.path.join(run_dir, "train_vec_norm.pkl"), VecMonitor(env))
    vn.training = False
    vn.norm_reward = False

    latents = []
    obs = vn.reset()
    while len(latents) < n_samples:
        with torch.no_grad():
            x = torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0)
            z = enc(x)
            z_q = torch.round(z / quant_step) * quant_step
            a = ctrl(z_q)
        latents.append(z_q.squeeze(0))
        action = np.clip(np.tanh(a.numpy()[0]), -1.0, 1.0)
        obs, _, done, _ = vn.step([action])
        if done[0]:
            obs = vn.reset()
    vn.close()

    Z = torch.stack(latents)
    with torch.no_grad():
        A = ctrl(Z)
    return Z, A


def fake_quant_per_tensor(x, bits=8):
    """STE fake quantization — matches ORT per-tensor symmetric."""
    qmax = 2**(bits-1) - 1
    scale = x.detach().abs().max() / qmax
    scale = scale.clamp(min=1e-8)
    x_q = (x / scale).round().clamp(-qmax, qmax) * scale
    return x + (x_q - x).detach()


def fake_quant_per_channel(w, bits=8):
    """STE fake quantization — matches ORT per-channel symmetric."""
    qmax = 2**(bits-1) - 1
    scale = w.detach().abs().max(dim=1, keepdim=True).values / qmax
    scale = scale.clamp(min=1e-8)
    w_q = (w / scale).round().clamp(-qmax, qmax) * scale
    return w + (w_q - w).detach()


def forward_with_fq(ctrl, x, fq_fraction=1.0):
    """Forward pass with fake quantization on weights and activations.
    fq_fraction in [0,1] blends clean and quantized paths for warmup."""
    modules = list(ctrl.modules())
    linears = [m for m in modules if isinstance(m, nn.Linear)]
    relu_after = set()
    prev_linear_idx = None
    for m in modules:
        if isinstance(m, nn.Linear):
            prev_linear_idx = id(m)
        elif isinstance(m, nn.ReLU) and prev_linear_idx is not None:
            relu_after.add(prev_linear_idx)

    h = x
    if fq_fraction > 0:
        h_fq = fake_quant_per_tensor(h)
        h = h * (1 - fq_fraction) + h_fq * fq_fraction

    for layer in linears:
        if fq_fraction > 0:
            w_fq = fake_quant_per_channel(layer.weight)
            w = layer.weight * (1 - fq_fraction) + w_fq * fq_fraction
        else:
            w = layer.weight
        h = F.linear(h, w, layer.bias)
        if id(layer) in relu_after:
            h = F.relu(h)
        if fq_fraction > 0:
            h_fq = fake_quant_per_tensor(h)
            h = h * (1 - fq_fraction) + h_fq * fq_fraction
    return h


def train_qat(ctrl, Z, A, epochs=100, bs=2048, lr=3e-4, warmup_epochs=10):
    opt = torch.optim.Adam(ctrl.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds = torch.utils.data.TensorDataset(Z, A)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    for ep in range(epochs):
        fq_frac = min(1.0, ep / max(warmup_epochs, 1))
        total = 0.0
        n = 0
        for zb, ab in dl:
            pred = forward_with_fq(ctrl, zb, fq_fraction=fq_frac)
            loss = F.mse_loss(pred, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * zb.size(0)
            n += zb.size(0)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}/{epochs}  loss={total/n:.6f}  fq={fq_frac:.2f}  lr={sched.get_last_lr()[0]:.2e}")


class CalibReader(CalibrationDataReader):
    def __init__(self, data, name, batch_size=1000):
        self.data = data.numpy()
        self.name = name
        self.bs = batch_size
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.data):
            return None
        batch = self.data[self.idx:self.idx + self.bs]
        self.idx += self.bs
        return {self.name: batch}


def export_and_quantize(ctrl, Z_calib, run_dir):
    ctrl.eval()
    n_latent = Z_calib.shape[1]
    onnx_f32 = os.path.join(run_dir, "latent_controller_qat.onnx")
    onnx_prep = os.path.join(run_dir, "latent_controller_qat_prep.onnx")
    onnx_int8 = os.path.join(run_dir, "latent_controller_qat_static_int8.onnx")

    torch.onnx.export(
        ctrl, torch.randn(1, n_latent), onnx_f32,
        input_names=["latent"], output_names=["action"],
        dynamic_axes={"latent": {0: "batch"}, "action": {0: "batch"}},
        opset_version=13,
    )
    quant_pre_process(onnx_f32, onnx_prep)

    sess = ort.InferenceSession(onnx_prep, providers=["CPUExecutionProvider"])
    reader = CalibReader(Z_calib[:20000], sess.get_inputs()[0].name)
    quantize_static(
        onnx_prep, onnx_int8, reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=True,
    )
    f32_sz = os.path.getsize(onnx_f32)
    i8_sz = os.path.getsize(onnx_int8)
    print(f"  f32: {f32_sz//1024} KB  INT8: {i8_sz//1024} KB  compression: {f32_sz/i8_sz:.2f}x")
    return onnx_f32, onnx_int8


def eval_reward(enc_onnx, ctrl_onnx, run_dir, env_name, quant_step, n_eps=20):
    def make_sess(p):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        return ort.InferenceSession(p, opts, providers=["CPUExecutionProvider"])

    enc = make_sess(enc_onnx)
    ctrl = make_sess(ctrl_onnx)

    env = DummyVecEnv([lambda: gym.make(env_name)])
    vn = VecNormalize.load(os.path.join(run_dir, "train_vec_norm.pkl"), VecMonitor(env))
    vn.training = False
    vn.norm_reward = False

    returns = []
    obs = vn.reset()
    ep_ret = 0.0
    while len(returns) < n_eps:
        x = obs[0].astype(np.float32)
        z = enc.run(None, {enc.get_inputs()[0].name: x})[0]
        z_q = np.round(z / quant_step) * quant_step
        a = ctrl.run(None, {ctrl.get_inputs()[0].name: z_q.reshape(1, -1)})[0][0]
        action = np.clip(np.tanh(a), -1.0, 1.0)
        obs, _, done, _ = vn.step([action])
        ep_ret += vn.get_original_reward()[0]
        if done[0]:
            returns.append(ep_ret)
            ep_ret = 0.0
            obs = vn.reset()
    vn.close()
    return np.array(returns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HalfCheetah-v4", choices=list(CONFIGS.keys()))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--n_samples", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=2048)
    args = ap.parse_args()

    cfg = CONFIGS[args.env]
    run_dir = cfg["run_dir"]
    quant_step = cfg["quant_step"]
    enc_onnx = os.path.join(run_dir, "encoder.onnx")

    print(f"=== QAT Training: {args.env} ===\n")

    print(f"Collecting {args.n_samples:,} samples...")
    Z, A = collect_data(run_dir, args.env, n_samples=args.n_samples, quant_step=quant_step)
    print(f"  Z: {Z.shape}  range: [{Z.min():.3f}, {Z.max():.3f}]")

    print(f"\nTraining ({args.epochs} epochs, {args.warmup} warmup)...")
    ctrl = copy.deepcopy(
        torch.load(os.path.join(run_dir, "latent_controller_full.pth"), weights_only=False).cpu().eval()
    )
    train_qat(ctrl, Z, A, epochs=args.epochs, bs=args.bs, lr=args.lr, warmup_epochs=args.warmup)

    torch.save(ctrl, os.path.join(run_dir, "latent_controller_qat_f32.pth"))

    print(f"\nWeight ranges after QAT:")
    for name, m in ctrl.named_modules():
        if hasattr(m, 'weight'):
            w = m.weight.data
            print(f"  {name}: [{w.min().item():.3f}, {w.max().item():.3f}]  absmax={w.abs().max().item():.3f}")

    print(f"\nExporting...")
    onnx_f32, onnx_int8 = export_and_quantize(ctrl, Z, run_dir)

    print(f"\nReward evaluation (20 episodes each):")
    for label, path in [("QAT f32", onnx_f32), ("QAT INT8", onnx_int8)]:
        rets = eval_reward(enc_onnx, path, run_dir, args.env, quant_step)
        print(f"  {label:<12} mean={rets.mean():8.1f}  std={rets.std():6.1f}  "
              f"min={rets.min():.1f}  max={rets.max():.1f}")


if __name__ == "__main__":
    main()

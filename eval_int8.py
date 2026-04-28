"""
eval_int8.py  —  Compare rollout performance of float32 vs INT8 latent controller.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")

import torch, torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

RD     = "sac_sweep_runs/HalfCheetah-v4/latent3/seed0"
N_EPS  = 20

def make_vn():
    env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
    vn  = VecNormalize.load(f"{RD}/train_vec_norm.pkl", VecMonitor(env))
    vn.training = False
    vn.norm_reward = False
    return vn

import copy

def quantize_weights_int8(ctrl):
    ctrl_q = copy.deepcopy(ctrl)
    for module in ctrl_q.modules():
        if isinstance(module, nn.Linear):
            w     = module.weight.data
            scale = w.abs().max(dim=1, keepdim=True).values / 127.0
            scale = scale.clamp(min=1e-8)
            w_int8 = (w / scale).round().clamp(-128, 127)
            module.weight.data = (w_int8 * scale).float()
    return ctrl_q

enc = torch.load(f"{RD}/encoder_full.pth",           weights_only=False).cpu().eval()
f32 = torch.load(f"{RD}/latent_controller_full.pth", weights_only=False).cpu().eval()
i8  = quantize_weights_int8(f32)


def run_policy(ctrl, label):
    vn = make_vn()
    returns = []
    obs = vn.reset()
    ep_ret = 0.0
    while len(returns) < N_EPS:
        with torch.no_grad():
            x  = torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0)
            z  = enc(x)
            a  = ctrl(z).numpy()[0]
        action = np.clip(np.tanh(a), -1.0, 1.0)
        obs, _, done, _ = vn.step([action])
        ep_ret += vn.get_original_reward()[0]
        if done[0]:
            returns.append(ep_ret)
            ep_ret = 0.0
            obs = vn.reset()
    vn.close()
    returns = np.array(returns)
    print(f"  {label:<10}  mean={returns.mean():8.1f}  std={returns.std():6.1f}"
          f"  min={returns.min():8.1f}  max={returns.max():8.1f}")
    return returns


print(f"\nHalfCheetah-v4  latent3  ({N_EPS} episodes each)")
print(f"  {'Model':<10}  {'mean':>8}  {'std':>6}  {'min':>8}  {'max':>8}")
print("  " + "-"*52)
r_f32 = run_policy(f32, "float32")
r_i8  = run_policy(i8,  "int8")

print(f"\n  Return drop (float32 → int8):  {r_f32.mean() - r_i8.mean():+.1f}")

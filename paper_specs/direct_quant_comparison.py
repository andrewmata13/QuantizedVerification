"""
direct_quant_comparison.py

Compute the number of quantized grid cells needed for direct observation-space
quantization (without a bottleneck) vs our 3-dim latent bottleneck approach
for HalfCheetah-v4. Also evaluates reward for the bottleneck policy.

Usage:
    python paper_specs/direct_quant_comparison.py
"""
import os, json

os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE_DIR = "sac_sweep_runs/HalfCheetah-v4/baseline/seed0"
BOTTLENECK_DIR = "sac_sweep_runs/HalfCheetah-v4/latent3/seed0"
QUANT_STEP = 0.05
DIRECT_QUANT_STEP = 0.1
N_LATENT = 3
N_EPISODES_ROLLOUT = 50
N_EPISODES_EVAL = 20


def collect_obs(run_dir, n_episodes):
    env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
    vn = VecNormalize.load(os.path.join(run_dir, "train_vec_norm.pkl"), VecMonitor(env))
    vn.training = False
    vn.norm_reward = False
    model = SAC.load(os.path.join(run_dir, "model.zip"), env=vn, device="cpu")

    obs_list = []
    obs = vn.reset()
    ep = 0
    while ep < n_episodes:
        act, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs[0].copy())
        obs, _, done, _ = vn.step(act)
        if done[0]:
            ep += 1
            obs = vn.reset()
    vn.close()
    return np.array(obs_list)


def eval_bottleneck_reward(run_dir, quant_step, n_episodes):
    env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
    vn = VecNormalize.load(os.path.join(run_dir, "train_vec_norm.pkl"), VecMonitor(env))
    vn.training = False
    vn.norm_reward = False

    encoder = torch.load(os.path.join(run_dir, "encoder_full.pth"), weights_only=False).cpu().eval()
    ctrl = torch.load(os.path.join(run_dir, "latent_controller_full.pth"), weights_only=False).cpu().eval()

    returns = []
    obs = vn.reset()
    ep_ret = 0.0
    while len(returns) < n_episodes:
        with torch.no_grad():
            z = encoder(torch.tensor(obs, dtype=torch.float32)).numpy()
            z_q = np.floor(z / quant_step) * quant_step + quant_step / 2.0
            a = ctrl(torch.tensor(z_q, dtype=torch.float32)).numpy()
        action = np.clip(np.tanh(a), -1.0, 1.0)
        obs, _, done, _ = vn.step(action)
        ep_ret += vn.get_original_reward()[0]
        if done[0]:
            returns.append(ep_ret)
            ep_ret = 0.0
            obs = vn.reset()
    vn.close()
    return np.array(returns)


def eval_baseline_reward(run_dir, n_episodes):
    env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
    vn = VecNormalize.load(os.path.join(run_dir, "train_vec_norm.pkl"), VecMonitor(env))
    vn.training = False
    vn.norm_reward = False
    model = SAC.load(os.path.join(run_dir, "model.zip"), env=vn, device="cpu")

    returns = []
    obs = vn.reset()
    ep_ret = 0.0
    while len(returns) < n_episodes:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = vn.step(act)
        ep_ret += vn.get_original_reward()[0]
        if done[0]:
            returns.append(ep_ret)
            ep_ret = 0.0
            obs = vn.reset()
    vn.close()
    return np.array(returns)


def eval_direct_quant_reward(run_dir, quant_step, n_episodes):
    env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
    vn = VecNormalize.load(os.path.join(run_dir, "train_vec_norm.pkl"), VecMonitor(env))
    vn.training = False
    vn.norm_reward = False
    model = SAC.load(os.path.join(run_dir, "model.zip"), env=vn, device="cpu")
    net = nn.Sequential(model.actor.latent_pi, model.actor.mu).cpu().eval()

    returns = []
    obs = vn.reset()
    ep_ret = 0.0
    while len(returns) < n_episodes:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32)
            obs_q = torch.floor(obs_t / quant_step) * quant_step + quant_step / 2.0
            a = net(obs_q).numpy()
        action = np.clip(np.tanh(a), -1.0, 1.0)
        obs, _, done, _ = vn.step(action)
        ep_ret += vn.get_original_reward()[0]
        if done[0]:
            returns.append(ep_ret)
            ep_ret = 0.0
            obs = vn.reset()
    vn.close()
    return np.array(returns)


def main():
    baseline_dir = os.path.join(BASE_DIR, BASELINE_DIR)
    bottleneck_dir = os.path.join(BASE_DIR, BOTTLENECK_DIR)

    print(f"Collecting {N_EPISODES_ROLLOUT} episodes ...")
    obs_arr = collect_obs(baseline_dir, N_EPISODES_ROLLOUT)

    obs_widths = obs_arr.max(axis=0) - obs_arr.min(axis=0)

    encoder = torch.load(os.path.join(bottleneck_dir, "encoder_full.pth"),
                         weights_only=False).cpu().eval()
    with torch.no_grad():
        z_all = encoder(torch.tensor(obs_arr, dtype=torch.float32)).numpy()
    z_widths = z_all.max(axis=0) - z_all.min(axis=0)
    cells_per_dim_latent = np.ceil(z_widths / QUANT_STEP).astype(int)
    total_cells_latent = int(np.prod(cells_per_dim_latent))
    print(f"  bottleneck quantization (step={QUANT_STEP}): {total_cells_latent} cells")

    print(f"\nEvaluating baseline (continuous) reward ({N_EPISODES_EVAL} episodes) ...")
    baseline_returns = eval_baseline_reward(baseline_dir, N_EPISODES_EVAL)
    baseline_mean = float(baseline_returns.mean())
    print(f"  mean: {baseline_mean:.1f}  std: {baseline_returns.std():.1f}")

    print(f"\nEvaluating direct quantized reward (step={DIRECT_QUANT_STEP}, {N_EPISODES_EVAL} episodes) ...")
    direct_returns = eval_direct_quant_reward(baseline_dir, DIRECT_QUANT_STEP, N_EPISODES_EVAL)
    direct_mean = float(direct_returns.mean())
    direct_pct = direct_mean / baseline_mean * 100
    print(f"  mean: {direct_mean:.1f}  std: {direct_returns.std():.1f}  ({direct_pct:.1f}% of baseline)")

    easy_frac, hard_frac = 0.10, 0.20
    easy_widths = obs_widths * easy_frac
    hard_widths = obs_widths * hard_frac

    def cells_for_widths(widths, step):
        return np.prod(np.ceil(widths / step).astype(int), dtype=float)

    direct_cells_full = f"{cells_for_widths(obs_widths, DIRECT_QUANT_STEP):.2e}"
    direct_cells_easy = f"{cells_for_widths(easy_widths, DIRECT_QUANT_STEP):.2e}"
    direct_cells_hard = f"{cells_for_widths(hard_widths, DIRECT_QUANT_STEP):.2e}"
    print(f"\n  Direct cells (step={DIRECT_QUANT_STEP}):")
    print(f"    full box:  {direct_cells_full}")
    print(f"    easy 10%:  {direct_cells_easy}")
    print(f"    hard 20%:  {direct_cells_hard}")

    print(f"\nEvaluating quantized bottleneck reward (step={QUANT_STEP}) ...")
    bottleneck_returns = eval_bottleneck_reward(bottleneck_dir, QUANT_STEP, N_EPISODES_EVAL)
    print(f"  mean: {bottleneck_returns.mean():.1f}  std: {bottleneck_returns.std():.1f}")

    result = {
        "n_obs": obs_arr.shape[1],
        "n_latent": N_LATENT,
        "baseline_reward_mean": round(baseline_mean, 1),
        "baseline_reward_std": round(float(baseline_returns.std()), 1),
        "direct_quant_step": DIRECT_QUANT_STEP,
        "direct_quant_reward_mean": round(direct_mean, 1),
        "direct_quant_reward_std": round(float(direct_returns.std()), 1),
        "direct_quant_pct_of_baseline": round(direct_pct, 1),
        "direct_total_cells_full_box": direct_cells_full,
        "direct_total_cells_easy_10pct": direct_cells_easy,
        "direct_total_cells_hard_20pct": direct_cells_hard,
        "bottleneck_quant_step": QUANT_STEP,
        "bottleneck_total_cells": total_cells_latent,
        "bottleneck_quant_reward_mean": round(float(bottleneck_returns.mean()), 1),
        "bottleneck_quant_reward_std": round(float(bottleneck_returns.std()), 1),
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "direct_quant_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()

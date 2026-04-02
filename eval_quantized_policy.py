"""
eval_quantized_policy.py

Compare episode return of the clean (unquantized) SAC bottleneck policy
against the same policy with the 1-D latent snapped to a quantization grid.

Quantization at inference:
    z_raw = encoder(obs)           # 1-D float
    z_q   = floor(z / q) * q + q/2  # snap to cell center
    action = tanh(latent_ctrl(z_q))

Usage:
    python eval_quantized_policy.py --env Hopper-v5
    python eval_quantized_policy.py --env HalfCheetah-v4 --quant_steps 0.5 1.0 2.0 4.0
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


def make_env(env_id, seed=0):
    def _thunk():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return _thunk


def load_policy(run_dir, env_id, seed=42):
    """Return (sac_model, encoder, latent_ctrl, vec_norm_env)."""
    env = DummyVecEnv([make_env(env_id, seed)])
    env = VecMonitor(env)

    vec_norm_path = os.path.join(run_dir, "train_vec_norm.pkl")
    eval_env = VecNormalize.load(vec_norm_path, env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = SAC.load(os.path.join(run_dir, "model.zip"), env=eval_env)

    encoder = torch.load(os.path.join(run_dir, "encoder_full.pth"),
                         weights_only=False).cpu().eval()
    latent_ctrl = torch.load(os.path.join(run_dir, "latent_controller_full.pth"),
                             weights_only=False).cpu().eval()
    return model, encoder, latent_ctrl, eval_env


def quantize(z: np.ndarray, quant_step: float) -> np.ndarray:
    """Snap each element of z to the center of its quantization cell."""
    cell_lower = np.floor(z / quant_step) * quant_step
    return cell_lower + quant_step / 2.0


def rollout(encoder, latent_ctrl, eval_env, n_episodes=20, quant_step=None, seed=42):
    """
    Run n_episodes and return array of episode returns.
    If quant_step is None, use the continuous (clean) latent.
    """
    tanh = nn.Tanh()
    returns = []

    obs = eval_env.reset()
    ep_ret = 0.0
    ep_count = 0

    while ep_count < n_episodes:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32)
            z = encoder(obs_t).numpy()          # shape (1, 1) or (1,)
            if quant_step is not None:
                z = quantize(z, quant_step)
            z_t = torch.tensor(z, dtype=torch.float32)
            action = tanh(latent_ctrl(z_t)).numpy()

        obs, reward, done, info = eval_env.step(action)
        ep_ret += reward[0]

        if done[0]:
            returns.append(ep_ret)
            ep_ret = 0.0
            ep_count += 1
            obs = eval_env.reset()

    return np.array(returns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env",         default="Hopper-v5",
                    choices=["Hopper-v5", "HalfCheetah-v4"])
    ap.add_argument("--run_dir",     default=None,
                    help="Path to run dir; defaults to sac_sweep_runs/<env>/arch0/seed0")
    ap.add_argument("--quant_steps", type=float, nargs="+",
                    default=[0.003, 0.005, 0.007, 0.01, 0.05, 0.1, 0.25],
                    help="Quantization step sizes to evaluate")
    ap.add_argument("--n_episodes",  type=int, default=20,
                    help="Episodes per configuration")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    run_dir = args.run_dir or f"sac_sweep_runs/{args.env}/arch0/seed0"

    print(f"Loading policy from {run_dir} ...")
    model, encoder, latent_ctrl, eval_env = load_policy(run_dir, args.env, args.seed)

    print(f"\nEvaluating over {args.n_episodes} episodes each\n")
    print(f"{'Config':<22}  {'Mean return':>12}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 65)

    # Clean (unquantized) baseline
    rets = rollout(encoder, latent_ctrl, eval_env,
                   n_episodes=args.n_episodes, quant_step=None, seed=args.seed)
    print(f"{'Clean (no quant)':<22}  {rets.mean():>12.1f}  {rets.std():>8.1f}"
          f"  {rets.min():>8.1f}  {rets.max():>8.1f}")

    # Quantized variants
    for qs in args.quant_steps:
        rets = rollout(encoder, latent_ctrl, eval_env,
                       n_episodes=args.n_episodes, quant_step=qs, seed=args.seed)
        label = f"quant_step={qs}"
        print(f"{label:<22}  {rets.mean():>12.1f}  {rets.std():>8.1f}"
              f"  {rets.min():>8.1f}  {rets.max():>8.1f}")

    eval_env.close()


if __name__ == "__main__":
    main()

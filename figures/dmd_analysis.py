"""
dmd_analysis.py

Dynamic Mode Decomposition (DMD) analysis of trained SAC policies.

Fits a global linear operator A: obs -> action via least squares on rollout data,
then computes SVD of A to reveal the dominant input-output modes. Compares the
DMD spectral decay to the Jacobian SVD decay from jacobian_analysis.py.

If the two spectra agree, it validates the latent dim choice from two independent
perspectives (global linear fit vs local linearization). If they differ, the
nonlinearity of the policy matters and the Jacobian is the right tool.

Usage:
    python figures/dmd_analysis.py
    python figures/dmd_analysis.py --n_obs 10000 --out figures/dmd_svd.png
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

CONFIGS = [
    {
        "env":     "HalfCheetah-v4",
        "run_dir": "sac_sweep_runs/HalfCheetah-v4/baseline/seed0",
        "color":   "#1976D2",
        "label":   "HalfCheetah-v4",
    },
    {
        "env":     "Hopper-v5",
        "run_dir": "sac_sweep_runs/Hopper-v5/baseline/seed0",
        "color":   "#388E3C",
        "label":   "Hopper-v5",
    },
]

RANK_THRESHOLD = 0.10


def collect_data(run_dir, env_id, n_obs):
    """Collect (normalized_obs, pre_tanh_action) pairs from deterministic rollout."""
    env_obj = DummyVecEnv([lambda: gym.make(env_id)])
    vn = VecNormalize.load(f"{run_dir}/train_vec_norm.pkl", VecMonitor(env_obj))
    vn.training = False; vn.norm_reward = False
    model = SAC.load(f"{run_dir}/model.zip", env=vn, device="cpu")

    # Extract pre-tanh action: actor.latent_pi -> actor.mu (no tanh)
    net = nn.Sequential(model.actor.latent_pi, model.actor.mu).cpu().eval()

    obs_list, act_list = [], []
    obs = vn.reset()
    while len(obs_list) < n_obs:
        obs_t = torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            act_pretanh = net(obs_t).numpy()[0]
        obs_list.append(obs[0].copy())
        act_list.append(act_pretanh.copy())
        act_tanh = np.tanh(act_pretanh)[np.newaxis]
        obs, _, done, _ = vn.step(act_tanh)
        if done[0]:
            obs = vn.reset()

    vn.close()
    return np.array(obs_list), np.array(act_list)


def fit_dmd(obs_arr, act_arr):
    """
    Fit global linear operator A: obs -> action via least squares.
    A = Y X^+ where X = obs.T, Y = act.T, X^+ is the pseudoinverse.

    Returns singular values of A, normalized to σ₁.
    """
    X = obs_arr.T   # [n_obs_dim, N]
    Y = act_arr.T   # [n_act_dim, N]

    # Thin SVD of X for stable pseudoinverse
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # Pseudoinverse: X^+ = V S^-1 U^T
    tol = s.max() * max(X.shape) * np.finfo(float).eps
    r = (s > tol).sum()
    X_pinv = Vt[:r].T @ np.diag(1.0 / s[:r]) @ U[:, :r].T   # [n_obs_dim, N]

    A = Y @ X_pinv   # [n_act_dim, n_obs_dim]

    sv = np.linalg.svd(A, compute_uv=False)
    sv_norm = sv / sv[0]
    return sv, sv_norm


def jacobian_svd(run_dir, env_id, n_samples=1000):
    """Compute mean Jacobian SVD over rollout states (same as jacobian_analysis.py)."""
    env_obj = DummyVecEnv([lambda: gym.make(env_id)])
    vn = VecNormalize.load(f"{run_dir}/train_vec_norm.pkl", VecMonitor(env_obj))
    vn.training = False; vn.norm_reward = False
    model = SAC.load(f"{run_dir}/model.zip", env=vn, device="cpu")
    net = nn.Sequential(model.actor.latent_pi, model.actor.mu).cpu().eval()

    n_obs = vn.observation_space.shape[0]
    n_act = vn.action_space.shape[0]

    obs_list = []
    obs = vn.reset()
    for _ in range(n_samples):
        act, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs[0].copy())
        obs, _, done, _ = vn.step(act)
        if done[0]: obs = vn.reset()
    vn.close()

    sv_list = []
    x = torch.tensor(np.array(obs_list), dtype=torch.float32)
    for i in range(n_samples):
        xi = x[i:i+1].requires_grad_(True)
        y = net(xi)
        J = torch.zeros(n_act, n_obs)
        for j in range(n_act):
            g = torch.autograd.grad(y[0, j], xi, retain_graph=True)[0]
            J[j] = g[0]
        sv = torch.linalg.svdvals(J).detach().numpy()
        sv_list.append(sv)

    mean_sv = np.stack(sv_list).mean(0)
    return mean_sv / mean_sv[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_obs",    type=int, default=5000,
                    help="Rollout observations for DMD fit (default: 5000)")
    ap.add_argument("--n_jac",    type=int, default=500,
                    help="States for Jacobian SVD (default: 500)")
    ap.add_argument("--out",      type=str, default="figures/dmd_svd.png")
    ap.add_argument("--threshold", type=float, default=RANK_THRESHOLD)
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)

    n_cols = len(CONFIGS)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    print(f"\n{'='*60}")
    print(f"DMD Analysis  (n_obs={args.n_obs}, n_jac={args.n_jac})")
    print(f"{'='*60}")

    for ax, cfg in zip(axes, CONFIGS):
        env_id  = cfg["env"]
        run_dir = cfg["run_dir"]
        color   = cfg["color"]

        print(f"\n--- {env_id} ---")

        # DMD
        print(f"  Collecting {args.n_obs} observations ...")
        obs_arr, act_arr = collect_data(run_dir, env_id, args.n_obs)
        sv_dmd, sv_dmd_norm = fit_dmd(obs_arr, act_arr)
        dmd_rank = int((sv_dmd_norm > args.threshold).sum())
        print(f"  DMD operator A ∈ R^{{{act_arr.shape[1]}×{obs_arr.shape[1]}}}")
        print(f"  DMD singular values (normalized):")
        for k, (sv, svn) in enumerate(zip(sv_dmd, sv_dmd_norm)):
            bar = "█" * int(svn * 25)
            print(f"    σ_{k+1} = {sv:.4f}  ({svn:.3f})  {bar}")
        print(f"  DMD effective rank (>{args.threshold}): {dmd_rank}")

        # Jacobian
        print(f"  Computing Jacobian SVD at {args.n_jac} states ...")
        sv_jac_norm = jacobian_svd(run_dir, env_id, args.n_jac)
        jac_rank = int((sv_jac_norm > args.threshold).sum())
        print(f"  Jacobian singular values (normalized):")
        for k, svn in enumerate(sv_jac_norm):
            bar = "█" * int(svn * 25)
            print(f"    σ_{k+1} = ({svn:.3f})  {bar}")
        print(f"  Jacobian effective rank (>{args.threshold}): {jac_rank}")

        # Agreement
        min_k = min(len(sv_dmd_norm), len(sv_jac_norm))
        corr = np.corrcoef(sv_dmd_norm[:min_k], sv_jac_norm[:min_k])[0, 1]
        print(f"  Pearson correlation DMD vs Jacobian spectra: {corr:.3f}")

        # Plot
        ks_dmd = np.arange(1, len(sv_dmd_norm) + 1)
        ks_jac = np.arange(1, len(sv_jac_norm) + 1)
        ax.bar(ks_dmd - 0.2, sv_dmd_norm, width=0.35, color=color, alpha=0.8,
               label=f"DMD (rank={dmd_rank})")
        ax.bar(ks_jac + 0.2, sv_jac_norm, width=0.35, color=color, alpha=0.4,
               label=f"Jacobian (rank={jac_rank})")
        ax.axhline(args.threshold, color="red", linestyle="--",
                   label=f"threshold ({args.threshold})")
        ax.set_xlabel("Singular value index k")
        ax.set_ylabel("σₖ / σ₁")
        ax.set_title(f"{env_id}\nDMD vs Jacobian SVD  (r={corr:.2f})")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {args.out}")


if __name__ == "__main__":
    main()

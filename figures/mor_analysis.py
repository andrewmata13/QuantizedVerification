"""
mor_analysis.py

Model order reduction (MOR) analysis: the Jacobian effective rank of the
baseline policy (σ_k/σ_1 > threshold) predicts the minimum latent dim N
needed to recover baseline performance.

One panel per environment. Each panel overlays:
  - Bar chart: σ_k / σ_1  (normalized singular values of mean Jacobian)
  - Markers:   R(N) = return_latentN / return_baseline  (right y-axis)
  - Dashed threshold line at σ_k/σ_1 = RANK_THRESHOLD
  - Vertical line at effective rank k* = |{k : σ_k/σ_1 > threshold}|

Key claim: k* (Jacobian effective rank) equals the smallest N where R(N) ≈ 1.

Caveat: Jacobian is bounded by min(n_obs, n_act) — Hopper max rank 3,
HalfCheetah max rank 6.

Usage:
    python figures/mor_analysis.py
    python figures/mor_analysis.py --n_samples 1000 --out figures/mor_analysis.png
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


# ── Configs ───────────────────────────────────────────────────────────────────
# Performance numbers taken from readme tables (mean return, deterministic eval).

RANK_THRESHOLD = 0.10   # σ_k/σ_1 > this → mode is "significant"

CONFIGS = [
    {
        "env":     "HalfCheetah-v4",
        "run_dir": "sac_sweep_runs/HalfCheetah-v4/baseline/seed0",
        "color":   "#1976D2",
        "baseline_return": 15264,
        "latent_returns": {   # latent_dim -> mean return
            1: 6683,
            2: 8705,
            3: 13522,
        },
    },
    {
        "env":     "Hopper-v5",
        "run_dir": "sac_sweep_runs/Hopper-v5/baseline/seed0",
        "color":   "#388E3C",
        "baseline_return": 3722,
        "latent_returns": {
            1: 1058,
            2: 976,
            3: 3538,
            4: 3587,
        },
    },
]


# ── Jacobian SVD ──────────────────────────────────────────────────────────────

def jacobian_singular_values(run_dir, env_id, n_samples):
    """Mean singular value spectrum of J(x) = ∂f/∂x over rollout states."""
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

    mean_sv = np.stack(sv_list).mean(0)   # [min(n_act, n_obs)]
    return mean_sv, n_obs, n_act


def cumulative_energy(mean_sv):
    """Cumulative fraction of squared-singular-value energy captured by top-k modes."""
    energy = mean_sv ** 2
    return np.cumsum(energy) / energy.sum()


# ── Plotting ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=500,
                    help="States for Jacobian SVD (default: 500)")
    ap.add_argument("--out",       type=str, default="figures/mor_analysis.png")
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)

    n_envs = len(CONFIGS)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5))
    if n_envs == 1:
        axes = [axes]

    print(f"\n{'='*60}")
    print(f"MOR Analysis (n_samples={args.n_samples})")
    print(f"{'='*60}")

    for ax, cfg in zip(axes, CONFIGS):
        env_id  = cfg["env"]
        run_dir = cfg["run_dir"]
        color   = cfg["color"]

        print(f"\n--- {env_id} ---")
        mean_sv, n_obs, n_act = jacobian_singular_values(run_dir, env_id, args.n_samples)
        max_k   = len(mean_sv)
        ks      = np.arange(1, max_k + 1)
        sv_n    = mean_sv / mean_sv[0]
        eff_rank = int((sv_n > RANK_THRESHOLD).sum())

        returns     = cfg["latent_returns"]
        base_r      = cfg["baseline_return"]
        latent_dims = sorted(returns.keys())
        retention   = np.array([returns[n] / base_r for n in latent_dims])

        print(f"  Jacobian: {n_act}×{n_obs}, max rank {max_k}, effective rank {eff_rank}")
        print(f"  σ_k/σ_1: {sv_n.round(3).tolist()}")
        print(f"  R(N):     {retention.round(3).tolist()}  (N={latent_dims})")

        # ── SVD bars (left y-axis) ────────────────────────────────────────
        bar_width = 0.35
        ks_f = np.array(ks, dtype=float)

        ax.bar(ks_f - bar_width / 2, sv_n, width=bar_width,
               color=color, alpha=0.85, label=r"$\sigma_k / \sigma_1$  (Jacobian SVD)")

        # threshold line
        ax.axhline(RANK_THRESHOLD, color=color, linestyle="--", linewidth=1.2, alpha=0.7,
                   label=f"threshold = {RANK_THRESHOLD}")

        # effective-rank marker
        ax.axvline(eff_rank + 0.5, color="black", linestyle=":", linewidth=1.5,
                   label=f"effective rank $k^*$ = {eff_rank}")

        ax.set_ylim(0, 1.15)
        ax.set_ylabel(r"$\sigma_k / \sigma_1$", color=color)
        ax.tick_params(axis="y", labelcolor=color)

        # ── Performance retention (right y-axis) ─────────────────────────
        ax2 = ax.twinx()
        ret_x = np.array(latent_dims, dtype=float)
        ax2.bar(ret_x + bar_width / 2, retention, width=bar_width,
                color="darkorange", alpha=0.75,
                label=r"$R(N)$ = return$_N$ / return$_\mathrm{base}$")
        ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax2.set_ylim(0, 1.35)
        ax2.set_ylabel("Fraction of baseline return $R(N)$", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")

        # Annotate retention bars
        for n, r, ret in zip(latent_dims, ret_x, retention):
            ax2.annotate(f"{returns[n]:,}", xy=(ret + bar_width / 2, r),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=8, color="darkorange")

        # ── Formatting ────────────────────────────────────────────────────
        xmax = max(max(latent_dims), max_k)
        ax.set_xticks(range(1, xmax + 1))
        ax.set_xlim(0.5, xmax + 0.5)
        ax.set_xlabel("Latent dim $N$ / singular value index $k$")
        ax.set_title(
            f"{env_id}\n"
            r"Blue bars: $\sigma_k/\sigma_1$   Orange bars: $R(N)$"
            f"\nJacobian $\\in \\mathbb{{R}}^{{{n_act}\\times{n_obs}}}$, "
            f"effective rank $k^*={eff_rank}$",
            fontsize=10,
        )
        ax.grid(True, alpha=0.25, axis="y")

        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    fig.suptitle(
        "Jacobian Effective Rank Predicts Minimum Latent Dim for Baseline Performance",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

"""
param_selection.py

Two-panel figure per environment showing the parameter selection rationale:
  Left  — Latent dim vs mean return (why we chose N=3)
  Right — Quant step vs mean return (why we chose q*)

The latent-dim panel uses hardcoded evaluation data (already collected).
The quant-step panel runs fresh rollouts over a sweep of step sizes.

Usage:
    python figures/param_selection.py
    python figures/param_selection.py --n_episodes 20 --out figures/param_selection.png
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


# ── Config ─────────────────────────────────────────────────────────────────────

CONFIGS = [
    {
        "env":            "HalfCheetah-v4",
        "color":          "#1976D2",
        "baseline_return": 15264,
        # latent_dim → (run_dir, clean_return)
        "latent_runs": {
            1: ("sac_sweep_runs/HalfCheetah-v4/arch0/seed0",   6683),
            2: ("sac_sweep_runs/HalfCheetah-v4/latent2/seed0", 8705),
            3: ("sac_sweep_runs/HalfCheetah-v4/latent3/seed0", 13522),
        },
        "chosen_latent":  3,
        # quant sweep: run_dir, chosen step, sweep values
        "quant_run_dir":  "sac_sweep_runs/HalfCheetah-v4/latent3/seed0",
        "chosen_quant":   0.05,
        "quant_steps":    [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    },
    {
        "env":            "Hopper-v5",
        "color":          "#388E3C",
        "baseline_return": 3722,
        "latent_runs": {
            1: ("sac_sweep_runs/Hopper-v5/arch0/seed0",   1058),
            2: ("sac_sweep_runs/Hopper-v5/latent2/seed0", 976),
            3: ("sac_sweep_runs/Hopper-v5/latent3/seed0", 3538),
            4: ("sac_sweep_runs/Hopper-v5/latent4/seed0", 3587),
        },
        "chosen_latent":  3,
        "quant_run_dir":  "sac_sweep_runs/Hopper-v5/latent3/seed0",
        "chosen_quant":   0.1,
        "quant_steps":    [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    },
]


# ── Quant-step evaluation ──────────────────────────────────────────────────────

def load_bottleneck(run_dir, env_id, seed=42):
    env = VecNormalize.load(
        f"{run_dir}/train_vec_norm.pkl",
        VecMonitor(DummyVecEnv([lambda: gym.make(env_id, render_mode=None)])),
    )
    env.training = False
    env.norm_reward = False
    env.seed(seed)
    encoder    = torch.load(f"{run_dir}/encoder_full.pth",
                            weights_only=False).cpu().eval()
    lat_ctrl   = torch.load(f"{run_dir}/latent_controller_full.pth",
                            weights_only=False).cpu().eval()
    return encoder, lat_ctrl, env


def quantize(z: np.ndarray, q: float) -> np.ndarray:
    return np.floor(z / q) * q + q / 2.0


def eval_quant(encoder, lat_ctrl, eval_env, quant_step, n_episodes, seed=42):
    tanh = nn.Tanh()
    returns, ep_ret = [], 0.0
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs = eval_env.reset()
    while len(returns) < n_episodes:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32)
            z = encoder(obs_t).numpy()
            if quant_step is not None:
                z = quantize(z, quant_step)
            action = tanh(lat_ctrl(torch.tensor(z, dtype=torch.float32))).numpy()
        obs, rew, done, _ = eval_env.step(action)
        ep_ret += rew[0]
        if done[0]:
            returns.append(ep_ret)
            ep_ret = 0.0
            obs = eval_env.reset()
    return np.array(returns)


def sweep_quant(cfg, n_episodes):
    run_dir  = cfg["quant_run_dir"]
    env_id   = cfg["env"]
    steps    = cfg["quant_steps"]

    print(f"  Loading {run_dir} ...")
    encoder, lat_ctrl, eval_env = load_bottleneck(run_dir, env_id)

    results = {}
    # clean (no quantization)
    r = eval_quant(encoder, lat_ctrl, eval_env, None, n_episodes, seed=42)
    results[None] = (r.mean(), r.std())
    print(f"    clean: {r.mean():.0f} ± {r.std():.0f}")

    for q in steps:
        r = eval_quant(encoder, lat_ctrl, eval_env, q, n_episodes, seed=42)
        results[q] = (r.mean(), r.std())
        marker = " ← chosen" if q == cfg["chosen_quant"] else ""
        print(f"    q={q:<6}: {r.mean():.0f} ± {r.std():.0f}{marker}")

    eval_env.close()
    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_latent_dim(ax, cfg):
    color      = cfg["color"]
    base_r     = cfg["baseline_return"]
    runs       = cfg["latent_runs"]
    chosen_n   = cfg["chosen_latent"]
    dims       = sorted(runs.keys())
    returns    = [runs[n][1] for n in dims]
    bar_colors = [color if n != chosen_n else "darkorange" for n in dims]

    bars = ax.bar(dims, returns, color=bar_colors, alpha=0.85, zorder=3)
    ax.axhline(base_r, color="black", linestyle="--", linewidth=1.5,
               label=f"baseline ({base_r:,})", zorder=2)

    # Annotate bars
    for d, r, b in zip(dims, returns, bars):
        ax.annotate(f"{r:,}", xy=(b.get_x() + b.get_width() / 2, r),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8)

    # Legend patch for chosen
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=color, alpha=0.85, label="latent controller"),
        Patch(facecolor="darkorange", alpha=0.85, label=f"chosen (N={chosen_n})"),
        plt.Line2D([0], [0], color="black", linestyle="--", label=f"baseline"),
    ], fontsize=8, loc="lower right")

    ax.set_xlabel("Latent dim $N$")
    ax.set_ylabel("Mean return")
    ax.set_title(f"{cfg['env']}  —  Latent dim vs return")
    ax.set_xticks(dims)
    ax.set_ylim(0, base_r * 1.18)
    ax.grid(True, alpha=0.25, axis="y", zorder=0)


def plot_quant_step(ax, cfg, quant_results):
    color      = cfg["color"]
    chosen_q   = cfg["chosen_quant"]
    steps      = cfg["quant_steps"]

    clean_mean, clean_std = quant_results[None]
    means = [quant_results[q][0] for q in steps]
    stds  = [quant_results[q][1] for q in steps]

    ax.axhline(clean_mean, color="black", linestyle="--", linewidth=1.5,
               label=f"clean (no quant): {clean_mean:.0f}", zorder=2)

    ax.plot(steps, means, "o-", color=color,
            linewidth=2, markersize=6, zorder=3,
            label="quantized return")

    # Highlight chosen
    chosen_mean = quant_results[chosen_q][0]
    ax.plot([chosen_q], [chosen_mean], "D", color="darkorange",
            markersize=10, zorder=4,
            label=f"chosen q={chosen_q} ({chosen_mean:.0f})")

    ax.set_xscale("log")
    ax.set_xlabel("Quantization step $q$")
    ax.set_ylabel("Mean return")
    ax.set_title(f"{cfg['env']}  —  Quant step vs return (latent{cfg['chosen_latent']})")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.25, zorder=0)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=20,
                    help="Episodes per quant-step config (default: 20)")
    ap.add_argument("--out", type=str, default="figures/param_selection.png")
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"\n{'='*60}")
    print(f"Parameter selection sweep (n_episodes={args.n_episodes})")
    print(f"{'='*60}")

    # Run quant sweeps first (slow part)
    all_quant = {}
    for cfg in CONFIGS:
        print(f"\n--- {cfg['env']} quant sweep ---")
        all_quant[cfg["env"]] = sweep_quant(cfg, args.n_episodes)

    # Plot
    n_envs = len(CONFIGS)
    fig, axes = plt.subplots(n_envs, 2, figsize=(12, 4.5 * n_envs))
    if n_envs == 1:
        axes = axes.reshape(1, 2)

    for row, cfg in enumerate(CONFIGS):
        plot_latent_dim(axes[row, 0], cfg)
        plot_quant_step(axes[row, 1], cfg, all_quant[cfg["env"]])

    fig.suptitle("Parameter Selection: Latent Dim and Quantization Step",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

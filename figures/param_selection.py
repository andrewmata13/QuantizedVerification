"""
param_selection.py

Two-panel figure showing parameter selection rationale across both environments:
  Left  — Latent dim vs % of baseline return (grouped bars, both envs)
  Right — Quant step vs % of baseline return (line, both envs, 85% threshold)

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

STYLE = os.path.join(os.path.dirname(__file__), "bak_matplotlib.mlpstyle")
THRESHOLD = 85.0   # % of baseline — used to pick chosen quant step

QUANT_STEPS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]

CONFIGS = [
    {
        "env":             "HalfCheetah-v4",
        "label":           "HalfCheetah",
        "color":           "#4488FF",
        "baseline_return": 15264,
        "latent_runs": {
            1: ("sac_sweep_runs/HalfCheetah-v4/arch0/seed0",   6683),
            2: ("sac_sweep_runs/HalfCheetah-v4/latent2/seed0", 8705),
            3: ("sac_sweep_runs/HalfCheetah-v4/latent3/seed0", 13522),
        },
        "chosen_latent":  3,
        "chosen_quant":   0.05,
        "quant_run_dir":  "sac_sweep_runs/HalfCheetah-v4/latent3/seed0",
        "quant_steps":    QUANT_STEPS,
    },
    {
        "env":             "Hopper-v5",
        "label":           "Hopper",
        "color":           "red",
        "baseline_return": 4117,
        "latent_runs": {
            1: ("sac_sweep_runs/Hopper-v5/arch0/seed0",   1058),
            2: ("sac_sweep_runs/Hopper-v5/latent2/seed0", 976),
            3: ("sac_sweep_runs/Hopper-v5/latent3/seed0", 3538),
            4: ("sac_sweep_runs/Hopper-v5/latent4/seed0", 3587),
        },
        "chosen_latent":  3,
        "chosen_quant":   0.1,
        "quant_run_dir":  "sac_sweep_runs/Hopper-v5/latent3/seed0",
        "quant_steps":    QUANT_STEPS,
    },
]


# ── Rollout helpers ────────────────────────────────────────────────────────────

def load_bottleneck(run_dir, env_id, seed=42):
    env = VecNormalize.load(
        f"{run_dir}/train_vec_norm.pkl",
        VecMonitor(DummyVecEnv([lambda: gym.make(env_id, render_mode=None)])),
    )
    env.training = False
    env.norm_reward = False
    env.seed(seed)
    encoder  = torch.load(f"{run_dir}/encoder_full.pth",
                          weights_only=False).cpu().eval()
    lat_ctrl = torch.load(f"{run_dir}/latent_controller_full.pth",
                          weights_only=False).cpu().eval()
    return encoder, lat_ctrl, env


def quantize(z, q):
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
    run_dir = cfg["quant_run_dir"]
    env_id  = cfg["env"]
    steps   = cfg["quant_steps"]
    print(f"  Loading {run_dir} ...")
    encoder, lat_ctrl, eval_env = load_bottleneck(run_dir, env_id)
    results = {}
    r = eval_quant(encoder, lat_ctrl, eval_env, None, n_episodes)
    results[None] = r.mean()
    print(f"    clean: {r.mean():.0f}")
    for q in steps:
        r = eval_quant(encoder, lat_ctrl, eval_env, q, n_episodes)
        results[q] = r.mean()
        print(f"    q={q}: {r.mean():.0f}")
    eval_env.close()
    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_latent_dim(ax, configs):
    width = 0.35
    # Exclude latent4 — only show dims present in all envs up to chosen
    all_dims = sorted({d for cfg in configs for d in cfg["latent_runs"]
                       if d <= cfg["chosen_latent"]})

    for i, cfg in enumerate(configs):
        base    = cfg["baseline_return"]
        runs    = cfg["latent_runs"]
        chosen  = cfg["chosen_latent"]
        color   = cfg["color"]
        offset  = (i - (len(configs) - 1) / 2) * width

        xs      = [d + offset for d in all_dims if d in runs]
        pcts    = [runs[d][1] / base * 100 for d in all_dims if d in runs]
        colors  = [color for d in all_dims if d in runs]

        ax.bar(xs, pcts, width=width * 0.9, color=colors,
               alpha=0.85, zorder=3, label=cfg["label"])

    ax.axhline(THRESHOLD, color="gray", linestyle=":", linewidth=1.2, zorder=2)

    ax.set_xticks(all_dims)
    ax.set_xticklabels([str(d) for d in all_dims])
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("% of Baseline Return")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.25, axis="y", zorder=0)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=cfg["color"], alpha=0.85, label=cfg["label"])
        for cfg in configs
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left")


def plot_quant_step(ax, configs, quant_results):
    for cfg in configs:
        base    = cfg["baseline_return"]
        steps   = cfg["quant_steps"]
        color   = cfg["color"]
        res     = quant_results[cfg["env"]]

        pcts = [res[q] / base * 100 for q in steps]

        ax.plot(steps, pcts, "o-", color=color, linewidth=2,
                markersize=5, zorder=3, label=cfg["label"])

        chosen_q   = cfg["chosen_quant"]
        chosen_pct = res[chosen_q] / base * 100
        ax.plot(chosen_q, chosen_pct, "D", color=color,
                markersize=9, markeredgecolor="black",
                markeredgewidth=2, zorder=4)

    ax.axhline(THRESHOLD, color="gray", linestyle=":", linewidth=1.5,
               label=f"{THRESHOLD:.0f}% threshold", zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Quantization Step")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([])
    ax.set_xticks(QUANT_STEPS)
    ax.set_xticklabels([str(q) for q in QUANT_STEPS])
    ax.grid(True, alpha=0.25, zorder=0)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=cfg["color"], marker="o", linewidth=2,
               markersize=5, label=cfg["label"])
        for cfg in configs
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower left")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--out", default="figures/param_selection.pdf")
    args = ap.parse_args()

    if os.path.exists(STYLE):
        plt.style.use(STYLE)

    os.makedirs("figures", exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"\nParameter selection sweep (n_episodes={args.n_episodes})")

    all_quant = {}
    for cfg in CONFIGS:
        print(f"\n--- {cfg['env']} ---")
        all_quant[cfg["env"]] = sweep_quant(cfg, args.n_episodes)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    plot_latent_dim(axes[0], CONFIGS)
    plot_quant_step(axes[1], CONFIGS, all_quant)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

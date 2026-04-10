"""
jacobian_analysis.py

Compute and plot the singular value distributions of the policy Jacobian for
HalfCheetah-v4 (baseline [256,256]) and Hopper-v5 (latent4 full network, ~baseline performance).

For each environment, J(x) = ∂f/∂x where f: obs -> pre-tanh action.
  HalfCheetah: J ∈ R^{6×17}, max rank 6  → shows effective rank ≈ 3 (explains latent3)
  Hopper:      J ∈ R^{3×11}, max rank 3  → output dim itself bounds rank;
               latent N < 3 is rank-deficient by construction (explains latent1/2 failure)

Usage:
    python figures/jacobian_analysis.py
    python figures/jacobian_analysis.py --n_samples 2000 --out figures/jacobian_svd.pdf
"""

import os, argparse
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


# ── Config ─────────────────────────────────────────────────────────────────────

ENVS = {
    "HalfCheetah-v4": {
        "run_dir":  "sac_sweep_runs/HalfCheetah-v4/baseline/seed0",
        "mode":     "baseline",
        "n_obs":    17,
        "n_act":    6,
        "label":    "HalfCheetah-v4\nbaseline [256,256]",
        "latent_returns": {1: 6683, 2: 8705, 3: 13522, "baseline": 14380},
        "color":    "#2196F3",
    },
    "Hopper-v5": {
        "run_dir":  "sac_sweep_runs/Hopper-v5/latent4/seed0",
        "mode":     "bottleneck",
        "n_obs":    11,
        "n_act":    3,
        "label":    "Hopper-v5\nlatent4 (return ≈ baseline)",
        "latent_returns": {1: 1058, 2: 1003, 3: 3538, 4: 3590},
        "color":    "#4CAF50",
    },
}


# ── Network loading ────────────────────────────────────────────────────────────

def load_network(env_id, cfg):
    rd = cfg["run_dir"]
    if cfg["mode"] == "baseline":
        env = DummyVecEnv([lambda: gym.make(env_id)])
        vn  = VecNormalize.load(f"{rd}/train_vec_norm.pkl", VecMonitor(env))
        vn.training = False; vn.norm_reward = False
        model = SAC.load(f"{rd}/model.zip", env=vn, device="cpu")
        net   = nn.Sequential(model.actor.latent_pi, model.actor.mu).cpu().eval()
        return net, vn
    else:  # bottleneck: use full encoder_full + latent_controller_full
        enc  = torch.load(f"{rd}/encoder_full.pth",           weights_only=False).cpu().eval()
        ctrl = torch.load(f"{rd}/latent_controller_full.pth", weights_only=False).cpu().eval()
        net  = nn.Sequential(enc, ctrl).eval()
        env  = DummyVecEnv([lambda: gym.make(env_id)])
        vn   = VecNormalize.load(f"{rd}/train_vec_norm.pkl", VecMonitor(env))
        vn.training = False; vn.norm_reward = False
        model = SAC.load(f"{rd}/model.zip", env=vn, device="cpu")
        return net, vn


def collect_obs(model, vn, n_steps):
    obs_list = []
    obs = vn.reset()
    for _ in range(n_steps):
        act, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs[0].copy())
        obs, _, done, _ = vn.step(act)
        if done[0]:
            obs = vn.reset()
    return np.array(obs_list)


# ── Jacobian computation ───────────────────────────────────────────────────────

def compute_singular_values(net, obs_arr, batch_size=256):
    """
    Return array of shape (N, min(n_out, n_in)) with singular values of J(x)
    for each observation x in obs_arr.
    """
    all_svs = []
    net.eval()
    for i in range(0, len(obs_arr), batch_size):
        batch = obs_arr[i:i+batch_size]
        svs_batch = []
        for x_np in batch:
            x = torch.tensor(x_np, dtype=torch.float32).requires_grad_(True)
            J = torch.autograd.functional.jacobian(net, x)   # (n_out, n_in)
            sv = torch.linalg.svdvals(J).detach().numpy()
            svs_batch.append(sv)
        all_svs.extend(svs_batch)
    return np.array(all_svs)   # (N, rank)


# ── Plotting ───────────────────────────────────────────────────────────────────

def make_figure(results, latent_returns_all, out_path):
    fig = plt.figure(figsize=(14, 10))

    # Layout: top row = two SVD violin plots, bottom = return vs latent dim
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    ax_hc  = fig.add_subplot(gs[0, 0])
    ax_hop = fig.add_subplot(gs[0, 1])
    ax_ret_hc  = fig.add_subplot(gs[1, 0])
    ax_ret_hop = fig.add_subplot(gs[1, 1])

    svd_axes  = [ax_hc,       ax_hop]
    ret_axes  = [ax_ret_hc,   ax_ret_hop]
    env_keys  = ["HalfCheetah-v4", "Hopper-v5"]

    for ax_svd, ax_ret, env_id in zip(svd_axes, ret_axes, env_keys):
        cfg   = ENVS[env_id]
        svs   = results[env_id]          # (N, rank)
        color = cfg["color"]
        rank  = svs.shape[1]

        # ── Normalized singular value bar chart (σₖ / σ₁) ───────────────────
        medians = np.median(svs, axis=0)
        q25     = np.percentile(svs, 25, axis=0)
        q75     = np.percentile(svs, 75, axis=0)
        norm    = medians / medians[0]         # normalise to σ₁ = 1
        norm_lo = (medians - q25) / medians[0]
        norm_hi = (q75 - medians) / medians[0]

        threshold = 0.10   # 10% of σ₁
        eff_rank  = int(np.sum(norm >= threshold))

        xlabels = [f"$\\sigma_{k+1}$" for k in range(rank)]
        bar_cols = [color if norm[k] >= threshold else "#BDBDBD" for k in range(rank)]

        bars = ax_svd.bar(xlabels, norm, color=bar_cols,
                          edgecolor="black", linewidth=0.8, width=0.55, zorder=3)
        ax_svd.errorbar(xlabels, norm,
                        yerr=[norm_lo, norm_hi],
                        fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)

        # 10% threshold line
        ax_svd.axhline(threshold, color="crimson", linestyle="--",
                       linewidth=1.5, zorder=5, label="10% threshold")

        # Annotate each bar with its absolute median
        for k, (bar, med) in enumerate(zip(bars, medians)):
            ax_svd.text(bar.get_x() + bar.get_width() / 2,
                        norm[k] + norm_hi[k] + 0.03,
                        f"{med:.2f}", ha="center", va="bottom",
                        fontsize=9, color="black")

        ax_svd.set_ylabel("Normalised singular value  ($\\sigma_k / \\sigma_1$)", fontsize=10)
        ax_svd.set_title(cfg["label"], fontsize=12, fontweight="bold")
        ax_svd.set_ylim(0, 1.35)
        ax_svd.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_svd.grid(axis="y", alpha=0.3, zorder=0)
        ax_svd.spines[["top", "right"]].set_visible(False)

        # Effective rank label
        ax_svd.text(0.97, threshold + 0.04,
                    f"Effective rank = {eff_rank}",
                    transform=ax_svd.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=9, color="crimson")

        # ── Return vs latent dim bar chart ────────────────────────────────────
        lr = cfg["latent_returns"]
        dims    = sorted([k for k in lr.keys() if isinstance(k, int)])
        returns = [lr[k] for k in dims]
        bar_colors = [color if k <= eff_rank else "#BDBDBD" for k in dims]

        bars = ax_ret.bar([str(k) for k in dims], returns,
                          color=bar_colors, edgecolor="black", linewidth=0.8)

        # Baseline line if present
        if "baseline" in lr:
            ax_ret.axhline(lr["baseline"], color="black", linestyle="--",
                           linewidth=1.5, label=f"Baseline ({lr['baseline']:,})")
            ax_ret.legend(fontsize=9)

        # Effective rank marker
        ax_ret.axvline(eff_rank - 0.5, color="red", linestyle="--", linewidth=1.5,
                       label=f"Eff. rank = {eff_rank}")

        # Labels
        for bar, ret in zip(bars, returns):
            ax_ret.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(returns) * 0.01,
                        f"{ret:,}", ha="center", va="bottom", fontsize=9)

        ax_ret.set_xlabel("Latent dimension N", fontsize=11)
        ax_ret.set_ylabel("Mean return (20 eps)", fontsize=11)
        ax_ret.set_title(f"{env_id.split('-')[0]} — Return vs Latent Dim", fontsize=12)
        ax_ret.set_ylim(0, max(returns) * 1.18)
        ax_ret.grid(axis="y", alpha=0.3)

        # Legend patches for coloring
        good_patch = mpatches.Patch(color=color,   label=f"N ≤ eff. rank ({eff_rank})")
        poor_patch = mpatches.Patch(color="#BDBDBD", label=f"N > eff. rank")
        ax_ret.legend(handles=[good_patch, poor_patch], fontsize=8, loc="lower right")

    fig.suptitle(
        "Policy Jacobian Singular Values and Latent Dimension Sufficiency\n"
        "Red dashed line = effective rank; bars shaded grey where N exceeds effective rank",
        fontsize=12, y=1.01,
    )

    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved {out_path}")
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=1000,
                    help="Number of rollout states to sample per env")
    ap.add_argument("--out", type=str, default="figures/jacobian_svd.png")
    args = ap.parse_args()

    results = {}

    for env_id, cfg in ENVS.items():
        print(f"\n=== {env_id} ===")
        net, vn = load_network(env_id, cfg)

        env_make = lambda eid=env_id: gym.make(eid)
        env = DummyVecEnv([env_make])
        vn2 = VecNormalize.load(f"{cfg['run_dir']}/train_vec_norm.pkl", VecMonitor(env))
        vn2.training = False; vn2.norm_reward = False
        model = SAC.load(f"{cfg['run_dir']}/model.zip", env=vn2, device="cpu")

        print(f"  Collecting {args.n_samples} states ...")
        obs_arr = collect_obs(model, vn2, args.n_samples)
        vn2.close()
        vn.close()

        print(f"  Computing Jacobians (J ∈ R^{{{cfg['n_act']}×{cfg['n_obs']}}}) ...")
        svs = compute_singular_values(net, obs_arr)
        results[env_id] = svs

        medians = np.median(svs, axis=0)
        print(f"  Median singular values: {np.round(medians, 3)}")
        threshold = medians[0] * 0.10
        eff_rank = int(np.sum(medians >= threshold))
        print(f"  Effective rank (>10% of σ₁): {eff_rank}")

    os.makedirs("figures", exist_ok=True)
    make_figure(results, None, args.out)


if __name__ == "__main__":
    main()

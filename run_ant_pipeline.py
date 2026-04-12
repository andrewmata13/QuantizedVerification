"""
run_ant_pipeline.py

Sequential pipeline for Ant-v4:
  1. Train baseline [256,256] for 3M steps
  2. Jacobian SVD analysis on converged baseline → effective rank
  3. Train latent(rank) and latent(rank+1) in parallel

Effective rank = number of singular values exceeding RANK_THRESHOLD * σ₁.
Based on HalfCheetah (eff.rank=3, J∈R^{6×17}) and Hopper (eff.rank=3, J∈R^{3×11}),
Ant-v4 (J∈R^{8×27}) is expected around 4–6.

Usage:
    python run_ant_pipeline.py
    python run_ant_pipeline.py --steps 3000000 --seed 0
    python run_ant_pipeline.py --skip_baseline   # skip if baseline already trained
    python run_ant_pipeline.py --latent_override 5   # skip Jacobian, use this rank
"""

import os, sys, time, argparse, subprocess
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

ENV        = "Ant-v4"
N_OBS      = 27
N_ACT      = 8
BASELINE_RD = "sac_sweep_runs/Ant-v4/baseline/seed0"
RANK_THRESHOLD = 0.10   # singular values > 10% of σ₁ count toward effective rank
N_JAC_SAMPLES  = 1000


# ── Phase 1: train baseline ────────────────────────────────────────────────────

def train(env, pi_arch, label, steps, seed):
    pi_str = " ".join(str(x) for x in pi_arch)
    cmd = [
        sys.executable, "train_sac.py",
        "--env",   env,
        "--pi",    *[str(x) for x in pi_arch],
        "--label", label,
        "--steps", str(steps),
        "--seed",  str(seed),
    ]
    env_vars = {**os.environ,
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MUJOCO_GL": "egl"}
    print(f"\n>>> Training {env} [{pi_str}] label={label} seed={seed} steps={steps}")
    print(f"    cmd: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env_vars)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  ERROR: training exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"  Done in {elapsed/3600:.2f}h")


# ── Phase 2: Jacobian effective rank ──────────────────────────────────────────

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


def jacobian_effective_rank(run_dir, n_samples=N_JAC_SAMPLES, threshold=RANK_THRESHOLD):
    """
    Load the baseline policy from run_dir, collect n_samples rollout observations,
    compute the Jacobian J = ∂f/∂x at each state, average the singular value
    spectrum, and return the effective rank.
    """
    env_obj = DummyVecEnv([lambda: gym.make(ENV)])
    vn = VecNormalize.load(f"{run_dir}/train_vec_norm.pkl", VecMonitor(env_obj))
    vn.training = False; vn.norm_reward = False
    model = SAC.load(f"{run_dir}/model.zip", env=vn, device="cpu")
    net = nn.Sequential(model.actor.latent_pi, model.actor.mu).cpu().eval()

    print(f"  Collecting {n_samples} rollout observations ...")
    obs_arr = collect_obs(model, vn, n_samples)
    vn.close()

    print(f"  Computing Jacobian at {n_samples} states ...")
    sv_list = []
    x = torch.tensor(obs_arr, dtype=torch.float32)
    for i in range(n_samples):
        xi = x[i:i+1].requires_grad_(True)
        y = net(xi)   # [1, N_ACT]
        J = torch.zeros(N_ACT, N_OBS)
        for j in range(N_ACT):
            g = torch.autograd.grad(y[0, j], xi, retain_graph=True)[0]
            J[j] = g[0]
        sv = torch.linalg.svdvals(J).detach().numpy()
        sv_list.append(sv)

    sv_arr = np.stack(sv_list)  # [n_samples, min(N_ACT, N_OBS)]
    mean_sv = sv_arr.mean(0)
    mean_sv_norm = mean_sv / mean_sv[0]

    print(f"\n  Ant-v4 baseline Jacobian (J ∈ R^{{{N_ACT}×{N_OBS}}})")
    print(f"  Mean singular values (normalised to σ₁):")
    for k, (sv, svn) in enumerate(zip(mean_sv, mean_sv_norm)):
        bar = "█" * int(svn * 30)
        flag = " ← rank cutoff" if abs(svn - threshold) < 0.02 else ""
        print(f"    σ_{k+1} = {sv:.4f}  ({svn:.3f})  {bar}{flag}")

    eff_rank = int((mean_sv_norm > threshold).sum())
    print(f"\n  Effective rank (σ_k / σ₁ > {threshold}): {eff_rank}")

    # Save SVD plot alongside the existing jacobian_svd.png
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ks = np.arange(1, len(mean_sv) + 1)
        ax.bar(ks, mean_sv_norm, color="#FF9800", alpha=0.8)
        ax.axhline(threshold, color="red", linestyle="--",
                   label=f"rank threshold ({threshold})")
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("σ_k / σ₁  (mean over rollout)")
        ax.set_title(f"Ant-v4 baseline Jacobian — effective rank = {eff_rank}")
        ax.legend()
        out = "figures/ant_jacobian_svd.png"
        os.makedirs("figures", exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved to {out}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    return eff_rank, mean_sv


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",           type=int,   default=3_000_000)
    ap.add_argument("--seed",            type=int,   default=0)
    ap.add_argument("--skip_baseline",   action="store_true",
                    help="Skip Phase 1 (baseline already trained).")
    ap.add_argument("--latent_override", type=int,   default=None,
                    help="Use this latent dim instead of Jacobian (skips Phase 2).")
    args = ap.parse_args()

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if not args.skip_baseline:
        print(f"\n{'='*60}")
        print(f"PHASE 1 — Train Ant-v4 baseline [256,256] ({args.steps/1e6:.0f}M steps)")
        print(f"{'='*60}")
        train(ENV, [256, 256], "baseline", args.steps, args.seed)
    else:
        print("Skipping Phase 1 (--skip_baseline).")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if args.latent_override is not None:
        eff_rank = args.latent_override
        print(f"\nSkipping Jacobian analysis, using latent_override={eff_rank}")
    else:
        print(f"\n{'='*60}")
        print(f"PHASE 2 — Jacobian effective rank analysis")
        print(f"{'='*60}")
        eff_rank, _ = jacobian_effective_rank(BASELINE_RD)

    latent_lo = eff_rank
    latent_hi = eff_rank + 1
    print(f"\n  → Chosen latent dims: {latent_lo} and {latent_hi}")

    # ── Phase 3: train latent_lo and latent_hi in parallel ────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 3 — Train latent{latent_lo} and latent{latent_hi} in parallel")
    print(f"{'='*60}")

    env_vars = {**os.environ,
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MUJOCO_GL": "egl"}

    procs = []
    for N in [latent_lo, latent_hi]:
        cmd = [
            sys.executable, "train_sac.py",
            "--env",   ENV,
            "--pi",    "16", str(N), "512", "512",
            "--label", f"latent{N}",
            "--steps", str(args.steps),
            "--seed",  str(args.seed),
        ]
        print(f"  Spawning: {' '.join(cmd)}")
        p = subprocess.Popen(cmd, env=env_vars)
        procs.append((N, p))

    # Wait for both
    for N, p in procs:
        p.wait()
        if p.returncode != 0:
            print(f"  ERROR: latent{N} training exited with code {p.returncode}")
        else:
            print(f"  latent{N} training complete.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Ant-v4 pipeline complete.")
    print(f"  Baseline : sac_sweep_runs/Ant-v4/baseline/seed{args.seed}/")
    print(f"  latent{latent_lo}  : sac_sweep_runs/Ant-v4/latent{latent_lo}/seed{args.seed}/")
    print(f"  latent{latent_hi}  : sac_sweep_runs/Ant-v4/latent{latent_hi}/seed{args.seed}/")
    print(f"\nNext steps:")
    print(f"  1. Evaluate: python eval_policy.py --env Ant-v4 (or check training logs)")
    print(f"  2. Generate specs: python generate_ant_specs.py  (to be written)")
    print(f"  3. Verify: python verify_policy.py --run_dir sac_sweep_runs/Ant-v4/latent{latent_lo}/seed0 --all")


if __name__ == "__main__":
    main()

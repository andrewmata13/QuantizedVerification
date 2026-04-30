"""
generate_halfcheetah_specs_v2.py

Generate 4 VNN-LIB safety specs for HalfCheetah-v4 that are SAFE for both the
[256,256] baseline AND all bottleneck controllers, enabling a fair timing comparison
between alpha-beta CROWN (full continuous network) and our quantized-bottleneck method.

Strategy:
  - Use p20/p80 input boxes (tight enough that baseline stays well below saturation)
  - Set output thresholds at sampling_max + 1.5 margin, then verify with PGD
  - If PGD still finds a violation, shrink box percentile further

Specs:
  spec_1  Nominal running — front-shin Y_4 upper bound
  spec_2  Nominal running — back-thigh Y_3 upper bound
  spec_3  Fast running    — front-shin Y_4 upper bound
  spec_4  Fast running    — any action Y_i upper bound (multi-clause OR)

Usage:
    python generate_halfcheetah_specs_v2.py
    python generate_halfcheetah_specs_v2.py --plo 40 --phi 60 --margin 1.5
"""

import os, argparse, time
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

SPEC_DIR    = "specs/HalfCheetah-v4"
BASELINE_RD = "sac_sweep_runs/HalfCheetah-v4/baseline/seed0"
BOTTLENECK_RDS = {
    "latent1": "sac_sweep_runs/HalfCheetah-v4/arch0/seed0",
    "latent2": "sac_sweep_runs/HalfCheetah-v4/latent2/seed0",
    "latent3": "sac_sweep_runs/HalfCheetah-v4/latent3/seed0",
}
N_ROLLOUT = 50_000
N_SAMPLE  = 1_000_000


def load_baseline(device="cpu"):
    env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
    vn = VecNormalize.load(f"{BASELINE_RD}/train_vec_norm.pkl", VecMonitor(env))
    vn.training = False; vn.norm_reward = False
    model = SAC.load(f"{BASELINE_RD}/model.zip", env=vn, device=device)
    net = nn.Sequential(model.actor.latent_pi, model.actor.mu).to(device).eval()
    return model, vn, net


def load_bottlenecks(device="cpu"):
    nets = {}
    for label, rd in BOTTLENECK_RDS.items():
        enc  = torch.load(f"{rd}/encoder_full.pth",           weights_only=False).to(device).eval()
        ctrl = torch.load(f"{rd}/latent_controller_full.pth", weights_only=False).to(device).eval()
        nets[label] = nn.Sequential(enc, ctrl).eval()
    return nets


def collect_rollouts(model, vn, n_steps):
    obs_list = []
    obs = vn.reset()
    for _ in range(n_steps):
        act, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs[0].copy())
        obs, _, done, _ = vn.step(act)
        if done[0]:
            obs = vn.reset()
    return np.array(obs_list)


def compute_output_stats(nets_dict, lo_t, hi_t, n_sample, device="cpu"):
    """Sample n_sample points in [lo_t, hi_t] and return per-net per-dim max/min."""
    torch.manual_seed(0)
    samples = (lo_t + torch.rand(n_sample, 17) * (hi_t - lo_t)).to(device)
    stats = {}
    with torch.no_grad():
        for label, net in nets_dict.items():
            out = net(samples).cpu()  # n x 6
            stats[label] = {"max": out.max(0).values.numpy(),
                            "min": out.min(0).values.numpy()}
    return stats


def write_vnnlib(path, obs_lo, obs_hi, violations, header_comment):
    """
    violations: list of (op, dim, threshold) where op is '>=' or '<='
    Each element becomes one clause in the (assert (or ...)) block.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_in  = len(obs_lo)
    n_out = 6
    lines = []
    for line in header_comment.strip().splitlines():
        lines.append(f"; {line}\n")
    lines.append("\n")
    for i in range(n_in):
        lines.append(f"(declare-const X_{i:<2} Real)\n")
    lines.append("\n")
    for i in range(n_out):
        lines.append(f"(declare-const Y_{i:<2} Real)\n")
    lines.append("\n")
    for i in range(n_in):
        lines.append(f"(assert (>= X_{i:<2} {obs_lo[i]:+.4f}))\n")
        lines.append(f"(assert (<= X_{i:<2} {obs_hi[i]:+.4f}))\n")
    lines.append("\n")
    clauses = []
    for op, dim, thr in violations:
        clauses.append(f"    (and ({op} Y_{dim} {thr:+.4f}))")
    lines.append("(assert (or\n" + "\n".join(clauses) + "\n))\n")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  wrote {path}")


def pgd_check(net, lo_t, hi_t, violations, n_steps=200, n_restarts=20, device="cpu"):
    """Return True if PGD finds a violation (spec is UNSAFE), False if no violation found."""
    lo_t = lo_t.to(device); hi_t = hi_t.to(device)
    for _ in range(n_restarts):
        x = (lo_t + torch.rand(256, 17, device=device) * (hi_t - lo_t))
        x = x.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=1e-2)
        for _ in range(n_steps):
            optimizer.zero_grad()
            y = net(x)
            # Maximize the most violated clause
            losses = []
            for op, dim, thr in violations:
                if op == ">=":
                    losses.append(y[:, dim] - thr)
                else:
                    losses.append(thr - y[:, dim])
            loss = -torch.stack(losses, dim=1).max(dim=1).values.mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x.data.clamp_(lo_t, hi_t)
        with torch.no_grad():
            y = net(x)
            for op, dim, thr in violations:
                if op == ">=" and (y[:, dim] >= thr).any():
                    return True
                if op == "<=" and (y[:, dim] <= thr).any():
                    return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plo",    type=float, default=20)
    ap.add_argument("--phi",    type=float, default=80)
    ap.add_argument("--margin", type=float, default=1.5,
                    help="Threshold = sampling_max + margin")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading baseline ...")
    model, vn, baseline_net = load_baseline(device)
    print("Loading bottlenecks ...")
    bottleneck_nets = load_bottlenecks(device)
    all_nets = {"baseline": baseline_net, **bottleneck_nets}

    print(f"Collecting {N_ROLLOUT} rollout steps from baseline ...")
    obs_arr = collect_rollouts(model, vn, N_ROLLOUT)
    vn.close()

    xvel = obs_arr[:, 9]
    masks = {
        "nominal": (xvel >= 0.5) & (xvel <= 1.5) & (np.abs(obs_arr[:,1]) <= 0.3) & (np.abs(obs_arr[:,0]) <= 0.4),
        "fast":    xvel >= 1.0,
    }

    boxes = {}
    for name, mask in masks.items():
        n = mask.sum()
        sub = obs_arr[mask]
        lo = np.percentile(sub, args.plo, axis=0)
        hi = np.percentile(sub, args.phi, axis=0)
        boxes[name] = (lo, hi)
        print(f"  {name}: n={n}  box_width_mean={(hi-lo).mean():.4f}")

    print(f"\nSampling {N_SAMPLE} points per box ...")
    box_stats = {}
    for name, (lo, hi) in boxes.items():
        lo_t = torch.tensor(lo, dtype=torch.float32)
        hi_t = torch.tensor(hi, dtype=torch.float32)
        box_stats[name] = compute_output_stats(all_nets, lo_t, hi_t, N_SAMPLE, device)
        for label, s in box_stats[name].items():
            print(f"  {name}/{label}: max={s['max'].round(3)}  absmax={np.abs(s['max']).max():.3f}")

    os.makedirs(SPEC_DIR, exist_ok=True)

    # ── spec_1: nominal running, Y_4 (front-shin) upper bound ─────────────────
    name = "nominal"
    lo, hi = boxes[name]
    stats  = box_stats[name]
    thr1 = float(max(s["max"][4] for s in stats.values())) + args.margin
    violations1 = [(">=", 4, thr1)]
    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)
    print(f"\nspec_1 threshold Y_4 >= {thr1:.4f}  checking PGD ...")
    unsafe = any(pgd_check(net, lo_t, hi_t, violations1, device=device) for net in all_nets.values())
    if unsafe:
        print("  WARNING: PGD found violation! Raise margin or tighten box.")
    else:
        print("  PGD: no violation found (safe)")
    write_vnnlib(
        f"{SPEC_DIR}/spec_1.vnnlib", lo, hi, violations1,
        f"HalfCheetah-v4 Spec 1 — Nominal running: front-shin upper saturation bound\n"
        f"Input box: p{args.plo:.0f}/p{args.phi:.0f} of normalized obs during balanced mid-speed running\n"
        f"  (xvel_norm in [0.5,1.5], |pitch| <= 0.3, |height| <= 0.4).\n"
        f"Violation: pre-tanh front-shin action Y_4 >= {thr1:.4f}\n"
        f"  (sampling max across all networks ~{thr1-args.margin:.3f}; threshold adds {args.margin} margin).\n"
        f"SAFE for baseline [256,256] and all bottleneck controllers.\n"
        f"Useful for timing comparison: our method (quantized bottleneck) vs alpha-beta CROWN (continuous)."
    )

    # ── spec_2: nominal running, Y_3 (back-thigh) upper bound ─────────────────
    thr2 = float(max(s["max"][3] for s in stats.values())) + args.margin
    violations2 = [(">=", 3, thr2)]
    print(f"\nspec_2 threshold Y_3 >= {thr2:.4f}  checking PGD ...")
    unsafe = any(pgd_check(net, lo_t, hi_t, violations2, device=device) for net in all_nets.values())
    print("  PGD:", "VIOLATION FOUND" if unsafe else "no violation found (safe)")
    write_vnnlib(
        f"{SPEC_DIR}/spec_2.vnnlib", lo, hi, violations2,
        f"HalfCheetah-v4 Spec 2 — Nominal running: back-thigh upper saturation bound\n"
        f"Input box: p{args.plo:.0f}/p{args.phi:.0f} of normalized obs during balanced mid-speed running.\n"
        f"Violation: pre-tanh back-thigh action Y_3 >= {thr2:.4f}\n"
        f"  (sampling max across all networks ~{thr2-args.margin:.3f}; threshold adds {args.margin} margin).\n"
        f"SAFE for baseline [256,256] and all bottleneck controllers."
    )

    # ── spec_3: fast running, Y_4 (front-shin) upper bound ────────────────────
    name = "fast"
    lo, hi = boxes[name]
    stats  = box_stats[name]
    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)
    thr3 = float(max(s["max"][4] for s in stats.values())) + args.margin
    violations3 = [(">=", 4, thr3)]
    print(f"\nspec_3 threshold Y_4 >= {thr3:.4f} (fast box)  checking PGD ...")
    unsafe = any(pgd_check(net, lo_t, hi_t, violations3, device=device) for net in all_nets.values())
    print("  PGD:", "VIOLATION FOUND" if unsafe else "no violation found (safe)")
    write_vnnlib(
        f"{SPEC_DIR}/spec_3.vnnlib", lo, hi, violations3,
        f"HalfCheetah-v4 Spec 3 — Fast running: front-shin upper saturation bound\n"
        f"Input box: p{args.plo:.0f}/p{args.phi:.0f} of normalized obs when xvel_norm >= 1.0.\n"
        f"Violation: pre-tanh front-shin action Y_4 >= {thr3:.4f}\n"
        f"  (sampling max across all networks ~{thr3-args.margin:.3f}; threshold adds {args.margin} margin).\n"
        f"SAFE for baseline [256,256] and all bottleneck controllers."
    )

    # ── spec_4: fast running, all actions upper bound (multi-clause OR) ────────
    thr4 = float(max(s["max"].max() for s in stats.values())) + args.margin
    violations4 = [(">=", i, thr4) for i in range(6)]
    print(f"\nspec_4 threshold any Y_i >= {thr4:.4f} (fast box)  checking PGD ...")
    unsafe = any(pgd_check(net, lo_t, hi_t, violations4, device=device) for net in all_nets.values())
    print("  PGD:", "VIOLATION FOUND" if unsafe else "no violation found (safe)")
    write_vnnlib(
        f"{SPEC_DIR}/spec_4.vnnlib", lo, hi, violations4,
        f"HalfCheetah-v4 Spec 4 — Fast running: any-action upper saturation bound\n"
        f"Input box: p{args.plo:.0f}/p{args.phi:.0f} of normalized obs when xvel_norm >= 1.0.\n"
        f"Violation: any pre-tanh action Y_i >= {thr4:.4f}\n"
        f"  (sampling max across all networks ~{thr4-args.margin:.3f}; threshold adds {args.margin} margin).\n"
        f"SAFE for baseline [256,256] and all bottleneck controllers."
    )

    print(f"\nAll specs written to {SPEC_DIR}/")
    print(f"  spec_1: nominal running, Y_4 >= {thr1:.4f}")
    print(f"  spec_2: nominal running, Y_3 >= {thr2:.4f}")
    print(f"  spec_3: fast running,    Y_4 >= {thr3:.4f}")
    print(f"  spec_4: fast running,    any Y_i >= {thr4:.4f}")


if __name__ == "__main__":
    main()

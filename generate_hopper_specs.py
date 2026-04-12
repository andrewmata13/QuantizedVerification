"""
generate_hopper_specs.py

Generate 4 VNN-LIB safety specs for Hopper-v5 that are SAFE for both the
[256,256] baseline AND all bottleneck controllers (latent1–4), enabling a
fair timing comparison between alpha-beta CROWN and our quantized-bottleneck method.

Hopper-v5 observation layout (11-dim, VecNormalize-normalized):
  X_0 : torso z-position (height)
  X_1 : torso pitch angle
  X_2 : thigh joint angle
  X_3 : leg joint angle
  X_4 : foot joint angle
  X_5 : x-velocity (forward)
  X_6 : z-velocity (vertical)
  X_7 : torso angular velocity
  X_8 : thigh joint angular velocity
  X_9 : leg joint angular velocity
  X_10: foot joint angular velocity

Hopper-v5 action layout (3-dim, pre-tanh):
  Y_0 : thigh joint torque
  Y_1 : leg joint torque
  Y_2 : foot joint torque

Strategy:
  - Collect baseline rollouts; filter for nominal balanced hopping
    (forward velocity in [0.5, 1.5] normalized, height and pitch bounded)
  - Use p20/p80 percentile input box — wide enough for interesting BaB work
  - Set output thresholds at sampling_max + 1.5 margin; verify with PGD
  - Each spec targets a different output dimension

Usage:
    python generate_hopper_specs.py
    python generate_hopper_specs.py --plo 20 --phi 80 --margin 1.5
"""

import os, argparse
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

SPEC_DIR    = "specs/Hopper-v5"
BASELINE_RD = "sac_sweep_runs/Hopper-v5/baseline/seed0"
BOTTLENECK_RDS = {
    "latent3": "sac_sweep_runs/Hopper-v5/latent3/seed0",
    "latent4": "sac_sweep_runs/Hopper-v5/latent4/seed0",
}
# latent1 (N=1) and latent2 (N=2) are excluded: these controllers are
# insufficiently expressive for Hopper (returns ~1000 vs baseline 3913)
# and produce unbounded outputs on the baseline observation distribution.
# Jacobian analysis confirms effective rank=3, so latent3/4 are the
# meaningful controllers to verify.
N_OBS     = 11
N_ACT     = 3
N_ROLLOUT = 50_000
N_SAMPLE  = 1_000_000


def load_baseline(device="cpu"):
    env = DummyVecEnv([lambda: gym.make("Hopper-v5")])
    vn  = VecNormalize.load(f"{BASELINE_RD}/train_vec_norm.pkl", VecMonitor(env))
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
    """Coarse random sampling — used only for initial insight; PGD max is authoritative."""
    torch.manual_seed(0)
    samples = (lo_t + torch.rand(n_sample, N_OBS) * (hi_t - lo_t)).to(device)
    stats = {}
    with torch.no_grad():
        for label, net in nets_dict.items():
            out = net(samples).cpu()
            stats[label] = {"max": out.max(0).values.numpy(),
                            "min": out.min(0).values.numpy()}
    return stats


def pgd_maximize(net, lo_t, hi_t, dim, n_steps=500, n_restarts=40, device="cpu"):
    """Find the maximum of output dim over the input box using PGD."""
    lo_t = lo_t.to(device); hi_t = hi_t.to(device)
    best = -np.inf
    for _ in range(n_restarts):
        x = (lo_t + torch.rand(256, N_OBS, device=device) * (hi_t - lo_t))
        x = x.detach().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=1e-2)
        for _ in range(n_steps):
            opt.zero_grad()
            loss = -net(x)[:, dim].mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.data.clamp_(lo_t, hi_t)
        with torch.no_grad():
            val = net(x)[:, dim].max().item()
            best = max(best, val)
    return best


def write_vnnlib(path, obs_lo, obs_hi, violations, header_comment):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    for line in header_comment.strip().splitlines():
        lines.append(f"; {line}\n")
    lines.append("\n")
    for i in range(N_OBS):
        lines.append(f"(declare-const X_{i:<2} Real)\n")
    lines.append("\n")
    for i in range(N_ACT):
        lines.append(f"(declare-const Y_{i:<2} Real)\n")
    lines.append("\n")
    for i in range(N_OBS):
        lines.append(f"(assert (>= X_{i:<2} {obs_lo[i]:+.4f}))\n")
        lines.append(f"(assert (<= X_{i:<2} {obs_hi[i]:+.4f}))\n")
    lines.append("\n")
    clauses = [f"    (and ({op} Y_{dim} {thr:+.4f}))" for op, dim, thr in violations]
    lines.append("(assert (or\n" + "\n".join(clauses) + "\n))\n")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  wrote {path}")


def pgd_check(net, lo_t, hi_t, violations, n_steps=200, n_restarts=20, device="cpu"):
    lo_t = lo_t.to(device); hi_t = hi_t.to(device)
    for _ in range(n_restarts):
        x = (lo_t + torch.rand(256, N_OBS, device=device) * (hi_t - lo_t))
        x = x.detach().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=1e-2)
        for _ in range(n_steps):
            opt.zero_grad()
            y = net(x)
            losses = []
            for op, dim, thr in violations:
                losses.append(y[:, dim] - thr if op == ">=" else thr - y[:, dim])
            loss = -torch.stack(losses, dim=1).max(dim=1).values.mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.data.clamp_(lo_t, hi_t)
        with torch.no_grad():
            y = net(x)
            for op, dim, thr in violations:
                vals = y[:, dim]
                if (op == ">=" and (vals >= thr).any()) or \
                   (op == "<=" and (vals <= thr).any()):
                    return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plo",    type=float, default=20)
    ap.add_argument("--phi",    type=float, default=80)
    ap.add_argument("--margin", type=float, default=1.5)
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

    # X_5 = forward velocity (normalized); X_0 = height; X_1 = pitch
    xvel  = obs_arr[:, 5]
    height = obs_arr[:, 0]
    pitch  = obs_arr[:, 1]

    print(f"  xvel:  [{xvel.min():.2f}, {xvel.max():.2f}]  mean={xvel.mean():.2f}")
    print(f"  height:[{height.min():.2f}, {height.max():.2f}]  mean={height.mean():.2f}")
    print(f"  pitch: [{pitch.min():.2f}, {pitch.max():.2f}]  mean={pitch.mean():.2f}")

    # Nominal balanced hopping: forward velocity in [0.5, 1.5], height and pitch bounded
    nominal_mask = (
        (xvel   >= 0.5) & (xvel   <= 1.5) &
        (np.abs(height) <= 1.5) &
        (np.abs(pitch)  <= 1.0)
    )
    n_nominal = nominal_mask.sum()
    print(f"  Nominal mask: {n_nominal}/{len(obs_arr)} steps ({100*n_nominal/len(obs_arr):.1f}%)")

    if n_nominal < 1000:
        print("  WARNING: too few nominal steps, using all steps")
        nominal_mask = np.ones(len(obs_arr), dtype=bool)

    sub = obs_arr[nominal_mask]
    lo  = np.percentile(sub, args.plo, axis=0)
    hi  = np.percentile(sub, args.phi, axis=0)
    print(f"  Box p{args.plo:.0f}/p{args.phi:.0f}: mean_width={(hi-lo).mean():.4f}")

    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)

    print(f"\nSampling {N_SAMPLE} points for output statistics ...")
    stats = compute_output_stats(all_nets, lo_t, hi_t, N_SAMPLE, device)
    for label, s in stats.items():
        print(f"  {label}: max={np.round(s['max'], 3)}")

    os.makedirs(SPEC_DIR, exist_ok=True)

    box_header = (
        f"Input box: p{args.plo:.0f}/p{args.phi:.0f} of normalized obs during "
        f"nominal balanced hopping\n"
        f"  (xvel_norm in [0.5,1.5], |height_norm| <= 1.5, |pitch_norm| <= 1.0).\n"
        f"SAFE for baseline [256,256] and bottleneck controllers latent3/latent4."
    )

    def make_spec(spec_id, dim, dim_name, action_name):
        print(f"\nspec_{spec_id}: PGD maximising Y_{dim} ({action_name}) over all networks ...")
        pgd_max = max(pgd_maximize(net, lo_t, hi_t, dim, device=device)
                      for net in all_nets.values())
        thr = pgd_max + args.margin
        print(f"  PGD max = {pgd_max:.4f}  →  threshold = {thr:.4f}")
        violations = [(">=", dim, thr)]
        print(f"  Verifying threshold with PGD check ...")
        unsafe = any(pgd_check(net, lo_t, hi_t, violations, device=device)
                     for net in all_nets.values())
        print(f"  PGD: {'VIOLATION FOUND' if unsafe else 'no violation found (safe)'}")
        write_vnnlib(
            f"{SPEC_DIR}/spec_{spec_id}.vnnlib", lo, hi, violations,
            f"Hopper-v5 — Spec {spec_id} — {dim_name} upper saturation bound\n"
            f"{box_header}\n"
            f"Violation: pre-tanh {action_name} Y_{dim} >= {thr:.4f}\n"
            f"  (sampling max across all networks ~{thr - args.margin:.4f}; "
            f"threshold adds {args.margin} margin)."
        )
        return thr, unsafe

    thr1, _ = make_spec(1, 0, "Thigh joint",  "thigh torque")
    thr2, _ = make_spec(2, 1, "Leg joint",    "leg torque")
    thr3, _ = make_spec(3, 2, "Foot joint",   "foot torque")

    # spec_4: all actions upper bound (OR clause) — threshold = max over all dims
    print(f"\nspec_4: PGD maximising each Y_i for OR-clause threshold ...")
    pgd_max4 = max(
        pgd_maximize(net, lo_t, hi_t, d, device=device)
        for net in all_nets.values() for d in range(N_ACT)
    )
    thr4 = pgd_max4 + args.margin
    print(f"  PGD max (all dims) = {pgd_max4:.4f}  →  threshold = {thr4:.4f}")
    violations4 = [(">=", i, thr4) for i in range(N_ACT)]
    print(f"  Verifying threshold with PGD check ...")
    unsafe = any(pgd_check(net, lo_t, hi_t, violations4, device=device)
                 for net in all_nets.values())
    print(f"  PGD: {'VIOLATION FOUND' if unsafe else 'no violation found (safe)'}")
    write_vnnlib(
        f"{SPEC_DIR}/spec_4.vnnlib", lo, hi, violations4,
        f"Hopper-v5 — Spec 4 — Any-action upper saturation bound\n"
        f"{box_header}\n"
        f"Violation: any pre-tanh action Y_i >= {thr4:.4f}\n"
        f"  (sampling max across all networks ~{thr4 - args.margin:.4f}; "
        f"threshold adds {args.margin} margin)."
    )

    print(f"\nAll specs written to {SPEC_DIR}/")
    print(f"  spec_1: Y_0 (thigh)  >= {thr1:.4f}")
    print(f"  spec_2: Y_1 (leg)    >= {thr2:.4f}")
    print(f"  spec_3: Y_2 (foot)   >= {thr3:.4f}")
    print(f"  spec_4: any Y_i      >= {thr4:.4f}")


if __name__ == "__main__":
    main()

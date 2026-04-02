"""
generate_halfcheetah_specs.py

Generate semantically grounded VNN-LIB specs for HalfCheetah-v4 by rolling out
the trained policy and computing per-behavior bounding boxes.

HalfCheetah-v4 normalized observation layout (17 dims):
  X_0  torso height      X_9  forward velocity (x)
  X_1  torso pitch       X_10 lateral velocity (y)
  X_2  torso roll        X_11 vertical velocity (z)
  X_3  back-thigh angle  X_12 back-thigh ang-vel
  X_4  back-shin angle   X_13 back-shin ang-vel
  X_5  back-foot angle   X_14 back-foot ang-vel
  X_6  front-thigh angle X_15 front-thigh ang-vel
  X_7  front-shin angle  X_16 front-foot ang-vel
  X_8  front-foot angle

All values are in VecNormalize normalized space.

Specs generated:
  spec_4  Cruising at high speed   — input tight around fast running; SAT expected
  spec_5  Slow / recovery phase    — low forward velocity; SAT expected
  spec_6  Pitch instability        — large forward pitch + low height; SAT expected
  spec_7  Nominal balanced running — tight box from rollout p10/p90; UNSAT expected

Usage:
    python generate_halfcheetah_specs.py [--run_dir ...] [--n_steps 20000]
"""

import os, argparse
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

OBS_LABELS = [
    "height", "pitch", "roll",
    "bthigh", "bshin", "bfoot",
    "fthigh", "fshin", "ffoot",
    "xvel", "yvel", "zvel",
    "bthigh_v", "bshin_v", "bfoot_v",
    "fthigh_v", "ffoot_v",
]

SPEC_DIR = "specs/HalfCheetah-v4"


def collect_rollouts(run_dir, env_id, n_steps, seed=42):
    env = DummyVecEnv([lambda: gym.make(env_id)])
    eval_env = VecNormalize.load(run_dir + "/train_vec_norm.pkl", VecMonitor(env))
    eval_env.training = False
    eval_env.norm_reward = False

    model = SAC.load(run_dir + "/model.zip", env=eval_env)
    encoder = torch.load(run_dir + "/encoder_full.pth", weights_only=False).cpu().eval()

    obs_list, lat_list = [], []
    obs = eval_env.reset()
    for _ in range(n_steps):
        with torch.no_grad():
            z = encoder(torch.tensor(obs, dtype=torch.float32)).item()
        act, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs[0].copy())
        lat_list.append(z)
        obs, _, done, _ = eval_env.step(act)
        if done[0]:
            obs = eval_env.reset()
    eval_env.close()
    return np.array(obs_list), np.array(lat_list)


def pct_box(obs, mask, plo=5, phi=95):
    """Return per-dim (lo, hi) at given percentiles for masked rows."""
    sub = obs[mask]
    lo = np.percentile(sub, plo, axis=0)
    hi = np.percentile(sub, phi, axis=0)
    return lo, hi


def write_spec(path, obs_lo, obs_hi, latent_violation, comment, violation_comment):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(obs_lo)
    lines = [f"; {line}\n" for line in comment.strip().splitlines()]
    lines.append("\n")
    for i in range(n):
        lines.append(f"(declare-const X_{i:<2} Real)\n")
    lines.append("(declare-const Y_0  Real)\n\n")
    for i in range(n):
        lines.append(f"(assert (>= X_{i:<2} {obs_lo[i]:+.4f}))\n")
        lines.append(f"(assert (<= X_{i:<2} {obs_hi[i]:+.4f}))\n")
        if i < n - 1:
            lines.append("\n")
    lines.append(f"\n; {violation_comment}\n")
    lo_v, hi_v = latent_violation
    clauses = []
    if hi_v is not None:
        clauses.append(f"    (and (>= Y_0 {hi_v:+.4f}))")
    if lo_v is not None:
        clauses.append(f"    (and (<= Y_0 {lo_v:+.4f}))")
    lines.append(f"(assert (or\n" + "\n".join(clauses) + "\n))\n")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="sac_sweep_runs/HalfCheetah-v4/arch0/seed0")
    ap.add_argument("--env_id",  default="HalfCheetah-v4")
    ap.add_argument("--n_steps", type=int, default=20000)
    args = ap.parse_args()

    print(f"Collecting {args.n_steps} rollout steps ...")
    obs, lats = collect_rollouts(args.run_dir, args.env_id, args.n_steps)
    xvel = obs[:, 9]   # normalized forward velocity

    print(f"Latent range: [{lats.min():.4f}, {lats.max():.4f}]  "
          f"mean={lats.mean():.4f}  std={lats.std():.4f}")

    # ── Behavioral masks (in normalized obs space) ─────────────────────────
    fast   = xvel >= 1.0               # fast forward running
    slow   = (xvel >= -0.3) & (xvel < 0.3)  # around average velocity
    pitch  = obs[:, 1] >= 0.8         # large forward pitch (instability)
    nominal = (
        (xvel >= 0.5) & (xvel <= 1.5) &
        (np.abs(obs[:, 1]) <= 0.3) &  # mild pitch
        (np.abs(obs[:, 0]) <= 0.4)    # normal height
    )

    for name, mask in [("fast", fast), ("slow", slow),
                       ("pitch_instab", pitch), ("nominal", nominal)]:
        n = mask.sum()
        if n == 0:
            print(f"  {name}: no samples")
            continue
        z = lats[mask]
        print(f"  {name:16s} n={n:5d}  "
              f"lat=[{z.min():.3f},{z.max():.3f}]  "
              f"p5={np.percentile(z,5):.3f}  p95={np.percentile(z,95):.3f}")

    # ── spec_4: High-speed cruising ────────────────────────────────────────
    # Fast forward running (xvel normalized ≥ 1.0).
    # Violation: latent escapes the observed fast-running range (> p99 + margin).
    # Expected SAT (wide input box; some extreme corners can push latent high).
    if fast.sum() > 50:
        lo, hi = pct_box(obs, fast, plo=2, phi=98)
        lat_hi = np.percentile(lats[fast], 99) + 0.5   # generous upper bound
        write_spec(
            f"{SPEC_DIR}/spec_4.vnnlib", lo, hi,
            latent_violation=(None, lat_hi),
            comment=f"""\
HalfCheetah-v4 Spec 4 — High-speed cruising
Input box: p2/p98 of normalized obs when xvel_norm >= 1.0 (fast running).
Violation: latent Y_0 >= {lat_hi:.4f}
  (p99 of fast-running latent + 0.5 margin)
SAT = encoder can produce a large latent for some fast-running input.
UNSAT = latent stays within {lat_hi:.4f} for all states in this box.""",
            violation_comment=f"Violation: latent exceeds fast-running range upper bound {lat_hi:.4f}",
        )

    # ── spec_5: Slow / recovery ────────────────────────────────────────────
    # Below-average forward velocity. Policy is in a different behavioral regime.
    # Violation: same upper bound — do slow states push the latent higher than fast ones?
    if slow.sum() > 50:
        lo, hi = pct_box(obs, slow, plo=2, phi=98)
        lat_hi_fast = np.percentile(lats[fast], 95) if fast.sum() > 0 else 1.0
        write_spec(
            f"{SPEC_DIR}/spec_5.vnnlib", lo, hi,
            latent_violation=(None, lat_hi_fast),
            comment=f"""\
HalfCheetah-v4 Spec 5 — Slow / recovery phase
Input box: p2/p98 of normalized obs when xvel_norm in [-0.3, 0.3].
Violation: latent Y_0 >= {lat_hi_fast:.4f}
  (p95 of fast-running latent — slow states shouldn't look like fast running).
SAT = some slow-phase input drives latent above fast-running threshold.
UNSAT = slow-phase latent always below fast-running regime.""",
            violation_comment=f"Violation: slow-phase latent exceeds fast-running threshold {lat_hi_fast:.4f}",
        )

    # ── spec_6: Pitch instability ──────────────────────────────────────────
    # Large forward pitch — cheetah is tumbling / falling forward.
    # Violation: latent > observed-max during stable running (p99 of full dataset).
    # SAT expected: unstable states can reach latent values far outside nominal range.
    if pitch.sum() > 50:
        lo, hi = pct_box(obs, pitch, plo=2, phi=98)
        lat_hi_nominal = np.percentile(lats[nominal], 99) + 0.1 if nominal.sum() > 0 else 1.0
        write_spec(
            f"{SPEC_DIR}/spec_6.vnnlib", lo, hi,
            latent_violation=(None, lat_hi_nominal),
            comment=f"""\
HalfCheetah-v4 Spec 6 — Pitch instability
Input box: p2/p98 of normalized obs when pitch_norm >= 0.8 (large forward tilt).
Violation: latent Y_0 >= {lat_hi_nominal:.4f}
  (p99 of nominal-running latent + 0.1 margin).
SAT = some pitch-unstable input drives latent above the nominal running range.
UNSAT = encoder keeps pitch-unstable states within the nominal latent range.""",
            violation_comment=f"Violation: pitch-instability latent exceeds nominal range {lat_hi_nominal:.4f}",
        )

    # ── spec_7: Nominal balanced running (tight) ───────────────────────────
    # Tight box around the most common balanced running state.
    # Violation: latent outside [p1, p99] of nominal latent range.
    # UNSAT expected: within this tight box the encoder should be well-behaved.
    if nominal.sum() > 100:
        lo, hi = pct_box(obs, nominal, plo=10, phi=90)
        lat_lo = np.percentile(lats[nominal], 1)
        lat_hi = np.percentile(lats[nominal], 99)
        write_spec(
            f"{SPEC_DIR}/spec_7.vnnlib", lo, hi,
            latent_violation=(lat_lo, lat_hi),
            comment=f"""\
HalfCheetah-v4 Spec 7 — Nominal balanced running (tight)
Input box: p10/p90 of normalized obs during balanced mid-speed running
  (xvel_norm in [0.5,1.5], |pitch| <= 0.3, |height| <= 0.4).
Violation: latent Y_0 outside [{lat_lo:.4f}, {lat_hi:.4f}]
  (p1/p99 of nominal latent during this behavior).
UNSAT = encoder always maps this tight nominal box to the expected latent range.
SAT   = some corner of this box escapes the latent bounds.""",
            violation_comment=f"Violation: latent outside nominal range [{lat_lo:.4f}, {lat_hi:.4f}]",
        )

    print("\nDone. New specs written to", SPEC_DIR)


if __name__ == "__main__":
    main()

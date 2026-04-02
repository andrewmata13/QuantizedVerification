"""
generate_hopper_specs.py

After training with train_custom_sb3.py, run this script to:
  1. Load the trained encoder and VecNormalize stats
  2. Roll out many episodes and record encoder outputs for healthy vs near-failure states
  3. Print the latent statistics and write calibrated VNN-LIB specs to specs/Hopper-v5/

Usage:
    python generate_hopper_specs.py --run_dir sac_sweep_runs/Hopper-v5/arch0/seed0
"""
import os, argparse
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

HEALTHY_HEIGHT_MIN = 0.7
HEALTHY_ANGLE_MAX = 0.2


def make_env(seed=0):
    def _thunk():
        env = gym.make("Hopper-v5")
        env.reset(seed=seed)
        return env
    return _thunk


def collect_latents(encoder, model, vec_norm, n_episodes=50):
    """Roll out n_episodes and collect (normalized_obs, latent, is_healthy) tuples."""
    healthy_latents = []
    failing_latents = []

    env = gym.make("Hopper-v5")

    for ep in range(n_episodes):
        obs_raw, _ = env.reset()
        done = False
        while not done:
            # Normalize obs the same way VecNormalize would
            obs_norm = vec_norm.normalize_obs(obs_raw[None])[0]

            # Compute latent
            with torch.no_grad():
                latent = encoder(torch.tensor(obs_norm, dtype=torch.float32)).item()

            # Classify state health (raw observation)
            height = obs_raw[0]
            angle  = obs_raw[1]
            is_healthy = (height > HEALTHY_HEIGHT_MIN) and (abs(angle) < HEALTHY_ANGLE_MAX)

            if is_healthy:
                healthy_latents.append((obs_norm, latent))
            else:
                failing_latents.append((obs_norm, latent))

            action, _ = model.predict(obs_norm[None], deterministic=True)
            obs_raw, _, terminated, truncated, _ = env.step(action[0])
            done = terminated or truncated

    env.close()
    return healthy_latents, failing_latents


def write_spec(path, header, obs_bounds, latent_violation):
    """
    Write a VNN-LIB spec file.

    obs_bounds: list of (lo, hi) for each of the 11 observations
    latent_violation: string with the (assert (or ...)) output constraint
    """
    lines = [header, ""]
    for i in range(11):
        lines.append(f"(declare-const X_{i}  Real)")
    lines.append("(declare-const Y_0 Real)")
    lines.append("")

    for i, (lo, hi) in enumerate(obs_bounds):
        lines.append(f"(assert (>= X_{i}  {lo:.4f}))")
        lines.append(f"(assert (<= X_{i}  {hi:.4f}))")

    lines.append("")
    lines.append(latent_violation)
    lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str,
                    default="sac_sweep_runs/Hopper-v5/arch0/seed0",
                    help="Path to a completed train_custom_sb3.py run directory")
    ap.add_argument("--n_episodes", type=int, default=50)
    args = ap.parse_args()

    run_dir = args.run_dir

    # Load encoder
    encoder_path = os.path.join(run_dir, "encoder_full.pth")
    encoder = torch.load(encoder_path, weights_only=False)
    encoder.eval()

    # Load SB3 model (for policy predictions during rollout)
    model_path = os.path.join(run_dir, "model.zip")
    model = SAC.load(model_path)

    # Load VecNormalize stats
    vec_norm_path = os.path.join(run_dir, "train_vec_norm.pkl")
    dummy_env = DummyVecEnv([make_env(seed=999)])
    vec_norm = VecNormalize.load(vec_norm_path, dummy_env)
    vec_norm.training = False
    vec_norm.norm_reward = False

    print(f"Collecting rollouts ({args.n_episodes} episodes)…")
    healthy, failing = collect_latents(encoder, model, vec_norm, n_episodes=args.n_episodes)

    h_vals = np.array([l for _, l in healthy])
    f_vals = np.array([l for _, l in failing]) if failing else np.array([])

    print(f"\n=== Latent statistics ===")
    print(f"Healthy states  : n={len(h_vals)}, "
          f"min={h_vals.min():.3f}, max={h_vals.max():.3f}, "
          f"mean={h_vals.mean():.3f}, std={h_vals.std():.3f}")
    if len(f_vals):
        print(f"Near-fail states: n={len(f_vals)}, "
              f"min={f_vals.min():.3f}, max={f_vals.max():.3f}, "
              f"mean={f_vals.mean():.3f}, std={f_vals.std():.3f}")
    else:
        print("Near-fail states: none encountered (well-trained agent!)")

    # Compute tight bounds: use 1st/99th percentile ± small margin
    margin = 0.5
    h_lo = float(np.percentile(h_vals, 1)) - margin
    h_hi = float(np.percentile(h_vals, 99)) + margin
    print(f"\nHealthy latent range (1–99 pct + {margin} margin): [{h_lo:.3f}, {h_hi:.3f}]")

    # Compute normalized obs bounds for healthy states
    h_obs = np.array([o for o, _ in healthy])
    obs_lo = h_obs.min(axis=0) - 0.2
    obs_hi = h_obs.max(axis=0) + 0.2

    # ---- Write calibrated spec_3.vnnlib (tight nominal separation) ----
    spec3_bounds = [(obs_lo[i], obs_hi[i]) for i in range(11)]

    # Violation = encoder escapes the healthy latent range
    violation = (
        f"(assert (or\n"
        f"    (and (>= Y_0  {h_hi:.4f}))\n"
        f"    (and (<= Y_0  {h_lo:.4f}))\n"
        f"))"
    )

    header3 = (
        f"; Hopper-v5 Spec 3 (auto-generated by generate_hopper_specs.py)\n"
        f"; Healthy latent range: [{h_lo:.3f}, {h_hi:.3f}]\n"
        f"; UNSAT proves encoder stays within this range for all nominal inputs."
    )

    write_spec(
        path="specs/Hopper-v5/spec_3_calibrated.vnnlib",
        header=header3,
        obs_bounds=spec3_bounds,
        latent_violation=violation,
    )

    # ---- Write spec_4.vnnlib: latent lower-bound for forward motion ----
    # Find the median latent for healthy states with high forward velocity
    if h_obs.shape[0] > 0:
        fwd_mask = h_obs[:, 5] > 0.5  # normalized forward velocity > 0.5
        if fwd_mask.sum() > 10:
            fwd_latents = h_vals[fwd_mask]
            fwd_lo = float(np.percentile(fwd_latents, 5)) - margin

            fwd_obs = h_obs[fwd_mask]
            fwd_obs_lo = fwd_obs.min(axis=0) - 0.1
            fwd_obs_hi = fwd_obs.max(axis=0) + 0.1

            # Force the forward-velocity bounds
            fwd_obs_lo[5] = 0.5
            fwd_obs_hi[5] = min(fwd_obs_hi[5], 4.0)

            violation4 = (
                f"(assert (or\n"
                f"    (and (<= Y_0  {fwd_lo:.4f}))\n"
                f"))"
            )
            header4 = (
                f"; Hopper-v5 Spec 4 (auto-generated by generate_hopper_specs.py)\n"
                f"; For healthy forward-moving states (norm fwd vel > 0.5),\n"
                f"; encoder latent should be >= {fwd_lo:.3f} (5th-pct − margin).\n"
                f"; UNSAT = encoder stays above this floor for all such inputs."
            )
            write_spec(
                path="specs/Hopper-v5/spec_4_forward_motion.vnnlib",
                header=header4,
                obs_bounds=[(fwd_obs_lo[i], fwd_obs_hi[i]) for i in range(11)],
                latent_violation=violation4,
            )

    dummy_env.close()
    print("\nDone. Update verify_hopper.py to use the calibrated specs.")


if __name__ == "__main__":
    main()

"""
train_sac.py

Unified SAC training script for any MuJoCo environment.
Supports bottleneck architectures (encoder/latent split) and standard baselines.

Usage:
    # Bottleneck — HalfCheetah latent dim 3
    python train_sac.py --env HalfCheetah-v4 --pi 16 3 512 512 --label latent3

    # Baseline — standard [256, 256]
    python train_sac.py --env HalfCheetah-v4 --pi 256 256 --label baseline

    # Walker, custom steps and seed
    python train_sac.py --env Walker2d-v4 --pi 16 3 512 512 --label latent3 --steps 5000000 --seed 1

    # Ant latent dim 4
    python train_sac.py --env Ant-v4 --pi 16 4 512 512 --label latent4

Output: sac_sweep_runs/<env>/<label>/seed<seed>/
  model.zip                   — full SB3 SAC model
  train_vec_norm.pkl          — VecNormalize stats
  full_network.onnx           — obs -> pre-tanh action (for alpha-beta CROWN)
  encoder.onnx                — obs -> latent (bottleneck only)
  encoder_full.pth            — encoder nn.Sequential (bottleneck only)
  latent_controller_full.pth  — latent -> action nn.Sequential (bottleneck only)

Bottleneck auto-detection: any pi layer with dim <= BOTTLENECK_THRESHOLD (default 32)
triggers the encoder/latent split. Standard arches like [256, 256] are treated as baseline.
"""

import os, time, argparse
os.environ["MUJOCO_GL"] = "egl"

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

BOTTLENECK_THRESHOLD = 32   # pi dims <= this trigger encoder/latent split
OUT_DIR = "sac_sweep_runs"


# ── Environment ────────────────────────────────────────────────────────────────

def make_env(env_id, seed):
    def _thunk():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return _thunk


# ── Bottleneck splitting ───────────────────────────────────────────────────────

def find_bottleneck_index(latent_pi):
    """Return index of the Linear layer with fewest output features."""
    min_dim, min_idx = 10**9, -1
    for i, layer in enumerate(latent_pi):
        if isinstance(layer, nn.Linear) and layer.out_features < min_dim:
            min_dim = layer.out_features
            min_idx = i
    return min_idx


def split_actor(model, latent_dim):
    """Split latent_pi at the bottleneck layer. Returns (encoder, latent_ctrl)."""
    latent_pi = model.actor.latent_pi
    for i, layer in enumerate(latent_pi):
        if isinstance(layer, nn.Linear) and layer.out_features == latent_dim:
            encoder    = nn.Sequential(*list(latent_pi.children())[:i + 2]).cpu().eval()
            latent_ctrl = nn.Sequential(
                nn.Sequential(*list(latent_pi.children())[i + 2:]),
                model.actor.mu
            ).cpu().eval()
            return encoder, latent_ctrl
    raise ValueError(f"No Linear layer with out_features={latent_dim} found in latent_pi")


# ── Checkpoint/resume ─────────────────────────────────────────────────────────

def find_latest_checkpoint(checkpoint_dir):
    """Return (path, steps_done) of latest checkpoint, or (None, 0)."""
    if not os.path.isdir(checkpoint_dir):
        return None, 0
    zips = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".zip"))
    if not zips:
        return None, 0
    latest = zips[-1]
    # SB3 filename: rl_model_{steps}_steps.zip
    try:
        steps_done = int(latest.split("_")[2])
    except (IndexError, ValueError):
        steps_done = 0
    return os.path.join(checkpoint_dir, latest), steps_done


# ── Export helpers ─────────────────────────────────────────────────────────────

def export_onnx(net, obs_dim, path, output_name="output"):
    dummy = torch.randn(obs_dim)
    with torch.no_grad():
        out = net(dummy)
    torch.onnx.export(
        net, dummy, path,
        input_names=["obs"], output_names=[output_name],
        opset_version=17, dynamo=False,
    )
    return out.shape[0]


# ── Main training function ─────────────────────────────────────────────────────

def train(env_id, pi_arch, qf_arch, label, seed, total_steps, eval_episodes):
    arch     = {"pi": pi_arch, "qf": qf_arch}
    run_dir  = os.path.join(OUT_DIR, env_id, label, f"seed{seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    is_bottleneck = min(pi_arch) <= BOTTLENECK_THRESHOLD
    bottleneck_dim = min(pi_arch) if is_bottleneck else None

    print(f"\n{'='*65}")
    print(f"  env:        {env_id}")
    print(f"  arch:       pi={pi_arch}  qf={qf_arch}")
    print(f"  label:      {label}  |  seed: {seed}  |  steps: {total_steps:,}")
    print(f"  bottleneck: {is_bottleneck}" + (f"  (dim={bottleneck_dim})" if is_bottleneck else ""))
    print(f"  run_dir:    {run_dir}")
    print(f"{'='*65}")

    # ── Build training env ────────────────────────────────────────────────────
    train_env = VecNormalize(
        VecMonitor(DummyVecEnv([make_env(env_id, seed)])),
        norm_obs=True, norm_reward=True, gamma=0.99,
    )

    # ── Resume or fresh start ─────────────────────────────────────────────────
    ckpt_path, steps_done = find_latest_checkpoint(ckpt_dir)
    if ckpt_path:
        print(f"  Resuming from {ckpt_path} ({steps_done:,} steps done)")
        model = SAC.load(ckpt_path, env=train_env, device="auto")
    else:
        model = SAC(
            "MlpPolicy", train_env,
            policy_kwargs=dict(net_arch=arch, activation_fn=nn.ReLU),
            learning_rate=3e-4, batch_size=256, tau=0.005, gamma=0.99,
            train_freq=(1, "step"), gradient_steps=1, target_entropy="auto",
            seed=seed, device="auto", verbose=1,
        )

    remaining = total_steps - steps_done
    if remaining <= 0:
        print(f"  Already complete ({steps_done:,} steps). Skipping training.")
    else:
        ckpt_cb = CheckpointCallback(
            save_freq=500_000, save_path=ckpt_dir,
            name_prefix="rl_model", save_vecnormalize=True,
        )
        t0 = time.time()
        model.learn(
            total_timesteps=remaining,
            callback=ckpt_cb,
            progress_bar=False,
            reset_num_timesteps=(steps_done == 0),
        )
        print(f"  Training done in {time.time() - t0:.0f}s")

    # ── Save model and norm stats ─────────────────────────────────────────────
    model.save(os.path.join(run_dir, "model.zip"))
    train_env.save(os.path.join(run_dir, "train_vec_norm.pkl"))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    eval_env = VecNormalize(
        VecMonitor(DummyVecEnv([make_env(env_id, seed + 123)])),
        training=False, norm_obs=True, norm_reward=False, gamma=0.99,
    )
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=eval_episodes, deterministic=True
    )
    print(f"  Eval ({eval_episodes} eps): mean={mean_r:.1f} ± {std_r:.1f}")

    # ── Export full network ONNX ──────────────────────────────────────────────
    obs_dim  = train_env.observation_space.shape[0]
    full_net = nn.Sequential(model.actor.latent_pi, model.actor.mu).cpu().eval()
    n_act    = export_onnx(full_net, obs_dim,
                           os.path.join(run_dir, "full_network.onnx"),
                           output_name="pre_tanh_action")
    print(f"  full_network.onnx  obs({obs_dim}) -> action({n_act})")

    # ── Bottleneck split + encoder ONNX ──────────────────────────────────────
    if is_bottleneck:
        encoder, latent_ctrl = split_actor(model, bottleneck_dim)
        torch.save(encoder,     os.path.join(run_dir, "encoder_full.pth"))
        torch.save(latent_ctrl, os.path.join(run_dir, "latent_controller_full.pth"))
        export_onnx(encoder, obs_dim,
                    os.path.join(run_dir, "encoder.onnx"),
                    output_name="latent")
        print(f"  encoder.onnx       obs({obs_dim}) -> latent({bottleneck_dim})")

    train_env.close()
    eval_env.close()
    return mean_r, std_r


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Train SAC on any MuJoCo env with optional bottleneck architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--env",    required=True,
                    help="Gymnasium env ID, e.g. HalfCheetah-v4, Walker2d-v4, Ant-v4")
    ap.add_argument("--pi",     type=int, nargs="+", required=True,
                    help="Policy network hidden layer sizes, e.g. --pi 16 3 512 512")
    ap.add_argument("--label",  type=str, default=None,
                    help="Run directory label (default: auto from pi arch)")
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--steps",  type=int, default=3_000_000,
                    dest="total_steps")
    ap.add_argument("--qf",     type=int, nargs="+", default=[256, 256],
                    help="Critic network hidden layer sizes (default: 256 256)")
    ap.add_argument("--eval_episodes", type=int, default=20)
    args = ap.parse_args()

    # Auto-derive label from arch if not given
    if args.label is None:
        bn_dim = min(args.pi)
        args.label = f"latent{bn_dim}" if bn_dim <= BOTTLENECK_THRESHOLD else "baseline"
        print(f"  (label auto-set to '{args.label}')")

    mean_r, std_r = train(
        env_id        = args.env,
        pi_arch       = args.pi,
        qf_arch       = args.qf,
        label         = args.label,
        seed          = args.seed,
        total_steps   = args.total_steps,
        eval_episodes = args.eval_episodes,
    )

    print(f"\n{'='*65}")
    print(f"  DONE  {args.env}/{args.label}/seed{args.seed}  "
          f"mean={mean_r:.1f} ± {std_r:.1f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

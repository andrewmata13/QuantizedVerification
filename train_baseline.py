"""
train_baseline.py

Train standard [256, 256] SAC (no bottleneck) for HalfCheetah-v4 and Hopper-v5.
Used as a performance baseline to quantify the cost of the bottleneck architecture.

Output dirs:
    sac_sweep_runs/HalfCheetah-v4/baseline/seed0/
    sac_sweep_runs/Hopper-v5/baseline/seed0/
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

SEED        = 0
OUT_DIR     = "sac_sweep_runs"
ARCH        = {"pi": [256, 256], "qf": [256, 256]}

ENV_CONFIG = {
    "HalfCheetah-v4": {"total_steps": 3_000_000},
    "Hopper-v5":      {"total_steps": 3_000_000},
}


def make_env(env_id, seed):
    def _thunk():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return _thunk


def train(env_id):
    cfg = ENV_CONFIG[env_id]
    run_dir = os.path.join(OUT_DIR, env_id, "baseline", f"seed{SEED}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Training {env_id}  arch={ARCH}  steps={cfg['total_steps']}")
    print(f"  run_dir: {run_dir}")
    print(f"{'='*60}")

    train_env = VecNormalize(
        VecMonitor(DummyVecEnv([make_env(env_id, SEED)])),
        norm_obs=True, norm_reward=True, gamma=0.99
    )
    model = SAC(
        "MlpPolicy", train_env,
        policy_kwargs=dict(net_arch=ARCH, activation_fn=nn.ReLU),
        learning_rate=3e-4, batch_size=256, tau=0.005, gamma=0.99,
        train_freq=(1, "step"), gradient_steps=1, target_entropy="auto",
        seed=SEED, device="auto", verbose=1,
    )
    # Resume from latest checkpoint if one exists
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")])
    steps_done = 0
    if checkpoints:
        latest = os.path.join(checkpoint_dir, checkpoints[-1])
        steps_done = int(checkpoints[-1].split("_")[2])
        print(f"  Resuming from checkpoint: {latest} ({steps_done} steps done)")
        model = SAC.load(latest, env=train_env, device="auto")
    remaining = cfg["total_steps"] - steps_done
    if remaining <= 0:
        print(f"  Already complete ({steps_done} steps). Skipping training.")
    else:
        checkpoint_cb = CheckpointCallback(
            save_freq=500_000, save_path=checkpoint_dir, name_prefix="rl_model",
            save_vecnormalize=True,
        )
        t0 = time.time()
        model.learn(total_timesteps=remaining, progress_bar=False,
                    callback=checkpoint_cb, reset_num_timesteps=(steps_done == 0))
        print(f"  Training done in {time.time()-t0:.0f}s")

    model.save(os.path.join(run_dir, "model.zip"))
    train_env.save(os.path.join(run_dir, "train_vec_norm.pkl"))

    eval_env = VecNormalize(
        VecMonitor(DummyVecEnv([make_env(env_id, SEED + 123)])),
        training=False, norm_obs=True, norm_reward=False, gamma=0.99
    )
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"  Eval (20 eps): mean={mean_r:.1f} ± {std_r:.1f}")

    # Export full network ONNX (actor.latent_pi + mu, no Tanh) for alpha-beta CROWN
    obs_dim = train_env.observation_space.shape[0]
    full_net = nn.Sequential(model.actor.latent_pi, model.actor.mu).cpu().eval()
    dummy = torch.randn(obs_dim)
    onnx_path = os.path.join(run_dir, "full_network.onnx")
    torch.onnx.export(
        full_net, dummy, onnx_path,
        input_names=["obs"], output_names=["pre_tanh_action"],
        opset_version=17, dynamo=False,
    )
    print(f"  ONNX saved: {onnx_path}  (obs({obs_dim}) -> action({full_net(dummy).shape[0]}))")

    train_env.close(); eval_env.close()
    return mean_r, std_r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", nargs="+",
                    default=["HalfCheetah-v4", "Hopper-v5"],
                    choices=list(ENV_CONFIG.keys()))
    args = ap.parse_args()

    results = []
    for env_id in args.env:
        mean_r, std_r = train(env_id)
        results.append((env_id, mean_r, std_r))

    print(f"\n{'='*60}")
    print("SUMMARY  arch=[256, 256]")
    print(f"{'='*60}")
    for env_id, mean_r, std_r in results:
        print(f"  {env_id:<20}  mean={mean_r:.1f} ± {std_r:.1f}")

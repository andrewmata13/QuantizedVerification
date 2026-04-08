"""
train_latent_sweep.py

Train HalfCheetah-v4 SAC with latent bottleneck dims 2 and 3 for verification
runtime comparison against the existing dim-1 model.

Architecture: obs(17) -> Linear(16) -> ReLU -> Linear(N) -> ReLU -> Linear(512) -> ReLU
              -> Linear(512) -> ReLU -> Linear(6) -> [tanh at runtime]

Output dirs:
    sac_sweep_runs/HalfCheetah-v4/latent2/seed0
    sac_sweep_runs/HalfCheetah-v4/latent3/seed0
"""

import os, time, json
os.environ["MUJOCO_GL"] = "egl"

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

ENV_ID     = "HalfCheetah-v4"
TOTAL_STEPS = 3_000_000
SEED        = 0
OUT_DIR     = "sac_sweep_runs"

LATENT_DIMS = [2, 3]


def make_env(seed=0):
    def _thunk():
        env = gym.make(ENV_ID)
        env.reset(seed=seed)
        return env
    return _thunk


def find_minimum_linear_index(model):
    min_dim, min_index = 1_000_000, -1
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear) and layer.out_features < min_dim:
            min_dim = layer.out_features
            min_index = i
    return min_index


def split_actor(model):
    actor_model = model.actor.latent_pi
    idx = find_minimum_linear_index(actor_model)
    print(f"  Bottleneck at layer {idx}: {actor_model[idx]}")
    encoder = nn.Sequential(*list(actor_model.children())[:idx + 2]).cpu()
    raw_latent = nn.Sequential(*list(actor_model.children())[idx + 2:])
    latent_ctrl = nn.Sequential(raw_latent, model.actor.mu).cpu()
    return encoder, latent_ctrl


def train(latent_dim):
    arch = {"pi": [16, latent_dim, 512, 512], "qf": [256, 256]}
    run_dir = os.path.join(OUT_DIR, ENV_ID, f"latent{latent_dim}", f"seed{SEED}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Training latent_dim={latent_dim}  arch={arch}  steps={TOTAL_STEPS}")
    print(f"  run_dir: {run_dir}")
    print(f"{'='*60}")

    train_env = VecNormalize(
        VecMonitor(DummyVecEnv([make_env(SEED)])),
        norm_obs=True, norm_reward=True, gamma=0.99
    )
    model = SAC(
        "MlpPolicy", train_env,
        policy_kwargs=dict(net_arch=arch, activation_fn=nn.ReLU),
        learning_rate=3e-4, batch_size=256, tau=0.005, gamma=0.99,
        train_freq=(1, "step"), gradient_steps=1, target_entropy="auto",
        seed=SEED, device="auto", verbose=1,
    )
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=False)
    print(f"  Training done in {time.time()-t0:.0f}s")

    model.save(os.path.join(run_dir, "model.zip"))
    train_env.save(os.path.join(run_dir, "train_vec_norm.pkl"))

    # Quick eval
    eval_env = VecNormalize(
        VecMonitor(DummyVecEnv([make_env(SEED + 123)])),
        training=False, norm_obs=True, norm_reward=False, gamma=0.99
    )
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"  Eval: mean={mean_r:.1f}  std={std_r:.1f}")

    # Split and export
    encoder, latent_ctrl = split_actor(model)
    encoder.eval(); latent_ctrl.eval()

    torch.save(encoder,     os.path.join(run_dir, "encoder_full.pth"))
    torch.save(latent_ctrl, os.path.join(run_dir, "latent_controller_full.pth"))

    obs_dim = train_env.observation_space.shape[0]
    torch.onnx.export(
        encoder, torch.randn(obs_dim),
        os.path.join(run_dir, "encoder.onnx"),
        input_names=["obs"], output_names=["latent"], opset_version=17,
        dynamo=False,
    )
    print(f"  Encoder ONNX saved  (obs({obs_dim}) -> latent({latent_dim}))")

    train_env.close(); eval_env.close()
    return mean_r


if __name__ == "__main__":
    for ld in LATENT_DIMS:
        mean_r = train(ld)
        print(f"\nlatent_dim={ld}  final mean return={mean_r:.1f}")

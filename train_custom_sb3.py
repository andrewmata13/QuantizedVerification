import os, time, json, csv
os.environ["MUJOCO_GL"] = "egl"

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

ENV_ID = "Hopper-v5"
TOTAL_STEPS = 5_000_000
SEEDS = [0]
NORMALIZE = True
DEVICE = "auto"

OUT_DIR = "sac_sweep_runs"

ARCHES = [
#    {"pi": [256, 256], "qf": [256, 256]},
#    {"pi": [6, 256, 256], "qf": [256, 256]},
#    {"pi": [5, 256, 256], "qf": [256, 256]},
#    {"pi": [4, 256, 256], "qf": [256, 256]},
#    {"pi": [3, 256, 256], "qf": [256, 256]},
#    {"pi": [2, 256, 256], "qf": [256, 256]},
#    {"pi": [3, 512, 512], "qf": [256, 256]},
    {"pi": [16, 1, 512, 512], "qf": [256, 256]},
#    {"pi": [8, 2, 256, 256], "qf": [256, 256]},
]

def make_env(env_id, seed):
    def _thunk():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return _thunk


def find_minimum_linear_index(model):
    """Find the index of the Linear layer with the fewest output features (the bottleneck)."""
    min_dim = 1_000_000
    min_index = -1
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            if layer.out_features < min_dim:
                min_dim = layer.out_features
                min_index = i
    return min_index


def split_sac_actor(model):
    """Split the SAC actor at its bottleneck layer into encoder and latent controller."""
    actor_model = model.actor.latent_pi
    min_layer_index = find_minimum_linear_index(actor_model)
    print(f"  Bottleneck at layer index {min_layer_index}: {actor_model[min_layer_index]}")
    encoder = nn.Sequential(*list(actor_model.children())[:min_layer_index + 2]).to("cpu")
    raw_latent = nn.Sequential(*list(actor_model.children())[min_layer_index + 2:])
    latent_controller = nn.Sequential(raw_latent, model.actor.mu).to("cpu")
    return encoder, latent_controller


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    t0 = time.time()

    for idx, arch in enumerate(ARCHES):
        seed_means = []

        for seed in SEEDS:
            run_dir = os.path.join(OUT_DIR, ENV_ID, f"arch{idx}", f"seed{seed}")
            os.makedirs(run_dir, exist_ok=True)

            train_env = DummyVecEnv([make_env(ENV_ID, seed)])
            train_env = VecMonitor(train_env)
            if NORMALIZE:
                train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, gamma=0.99)

            model = SAC(
                "MlpPolicy",
                train_env,
                policy_kwargs=dict(net_arch=arch, activation_fn=nn.ReLU),
                learning_rate=3e-4,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                target_entropy="auto",
                seed=seed,
                device=DEVICE,
                verbose=1,
            )

            model.learn(total_timesteps=TOTAL_STEPS, progress_bar=False)

            eval_env = DummyVecEnv([make_env(ENV_ID, seed + 123)])
            eval_env = VecMonitor(eval_env)

            # Save model and VecNormalize stats
            model_path = os.path.join(run_dir, "model.zip")
            model.save(model_path)

            if NORMALIZE:
                vec_norm_path = os.path.join(run_dir, "train_vec_norm.pkl")
                train_env.save(vec_norm_path)
                eval_vec = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, gamma=0.99)
                eval_vec.obs_rms, eval_vec.ret_rms = train_env.obs_rms, train_env.ret_rms
                eval_env_for_eval = eval_vec
            else:
                eval_env_for_eval = eval_env

            mean_r, std_r = evaluate_policy(
                model, eval_env_for_eval, n_eval_episodes=10, deterministic=True, render=False
            )
            seed_means.append(float(mean_r))

            # Split at bottleneck and export encoder to ONNX
            print(f"\nSplitting model at bottleneck for arch {arch}...")
            encoder, latent_controller = split_sac_actor(model)
            encoder.eval()
            latent_controller.eval()

            obs_dim = train_env.observation_space.shape[0]
            example_input = torch.randn(obs_dim)

            torch.save(encoder, os.path.join(run_dir, "encoder_full.pth"))
            torch.save(latent_controller, os.path.join(run_dir, "latent_controller_full.pth"))

            onnx_path = os.path.join(run_dir, "encoder.onnx")
            torch.onnx.export(
                encoder,
                example_input,
                onnx_path,
                input_names=["obs"],
                output_names=["latent"],
                opset_version=17,
            )
            print(f"  Encoder ONNX saved to {onnx_path}")
            print(f"  Encoder architecture: obs({obs_dim}) -> {arch['pi'][:arch['pi'].index(min(arch['pi']))+1]}")

            results.append({
                "env_id": ENV_ID,
                "arch": json.dumps(arch, separators=(",", ":")),
                "seed": seed,
                "mean": float(mean_r),
                "std": float(std_r),
                "run_dir": run_dir,
            })

            train_env.close()
            if NORMALIZE:
                eval_env_for_eval.close()
            else:
                eval_env.close()

        avg_mean = sum(seed_means) / len(seed_means)
        print(f"[{ENV_ID}] arch#{idx} {arch}  →  mean={avg_mean:.2f} (avg over {len(SEEDS)} seeds)")

    agg = {}
    for r in results:
        agg.setdefault(r["arch"], []).append(r["mean"])
    ranking = sorted(((k, sum(v)/len(v)) for k, v in agg.items()), key=lambda t: t[1], reverse=True)

    print("\n=== Sweep summary (avg over seeds) ===")
    for rank, (arch_key, avg_mean) in enumerate(ranking, 1):
        print(f"{rank:2d}. {arch_key}  →  mean={avg_mean:.2f}")

    print(f"\nDone in {time.time()-t0:.1f}s")

    # Save CSV summary
    csv_path = os.path.join(OUT_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["env_id", "arch", "seed", "mean", "std", "run_dir"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()

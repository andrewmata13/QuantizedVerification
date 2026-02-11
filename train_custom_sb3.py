import os, time, json, csv
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

ENV_ID = "Hopper-v5"
TOTAL_STEPS = 1000
SEEDS = [0]
NORMALIZE = True
DEVICE = "auto"

ARCHES = [
    {"pi": [256, 256], "qf": [256, 256]},
    {"pi": [6, 256, 256], "qf": [256, 256]},
    {"pi": [5, 256, 256], "qf": [256, 256]},
    {"pi": [4, 256, 256], "qf": [256, 256]},
    {"pi": [3, 256, 256], "qf": [256, 256]},
    {"pi": [2, 256, 256], "qf": [256, 256]},
    {"pi": [3, 256, 256, 256], "qf": [256, 256]},
    {"pi": [3, 512, 512], "qf": [256, 256]},
    {"pi": [11, 3, 512, 512], "qf": [256, 256]},
]

def make_env(env_id, seed):
    def _thunk():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return _thunk

def main():
    results = []
    t0 = time.time()

    for idx, arch in enumerate(ARCHES):
        seed_means = []

        for seed in SEEDS:
            train_env = DummyVecEnv([make_env(ENV_ID, seed)])
            train_env = VecMonitor(train_env)
            if NORMALIZE:
                train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, gamma=0.99)

            model = SAC(
                "MlpPolicy",
                train_env,
                policy_kwargs=dict(net_arch=arch),
                learning_rate=3e-4,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                target_entropy="auto",
                seed=seed,
                device=DEVICE,
                verbose=0,
            )

            model.learn(total_timesteps=TOTAL_STEPS, progress_bar=False)

            eval_env = DummyVecEnv([make_env(ENV_ID, seed + 123)])
            eval_env = VecMonitor(eval_env)
            if NORMALIZE:
                eval_vec = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, gamma=0.99)
                eval_vec.obs_rms, eval_vec.ret_rms = train_env.obs_rms, train_env.ret_rms
                eval_env_for_eval = eval_vec
            else:
                eval_env_for_eval = eval_env

            mean_r, std_r = evaluate_policy(
                model, eval_env_for_eval, n_eval_episodes=10, deterministic=True, render=False
            )
            seed_means.append(float(mean_r))

            results.append({
                "env_id": ENV_ID,
                "arch": json.dumps(arch, separators=(",", ":")),
                "seed": seed,
                "mean": float(mean_r),
                "std": float(std_r)
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

if __name__ == "__main__":
    main()

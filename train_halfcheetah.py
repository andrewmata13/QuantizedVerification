import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy


def make_env(env_id, seed=0):
    def _thunk():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk


def build_algo(algo, env, seed, lr, device, gamma, policy_kwargs=None):
    algo = algo.lower()
    if algo == "ppo":
        return PPO("MlpPolicy", env, verbose=1, seed=seed, learning_rate=lr,
                   policy_kwargs=policy_kwargs, device=device, gamma=gamma)
    if algo == "sac":
        return SAC("MlpPolicy", env, verbose=1, seed=seed, learning_rate=lr,
                   policy_kwargs=policy_kwargs, device=device, gamma=gamma)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="HalfCheetah-v4")
    ap.add_argument("--algo", type=str, default="sac", choices=["ppo", "sac"])
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=100_000)
    ap.add_argument("--eval_episodes", type=int, default=10)
    ap.add_argument("--save_dir", type=str, default="runs/halfcheetah_bestmodel")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--no_reward_norm", action="store_true")
    ap.add_argument("--no_obs_norm", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--load_path", type=str, default="")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "best").mkdir(parents=True, exist_ok=True)
    (save_dir / "eval").mkdir(parents=True, exist_ok=True)

    vn_gamma = 0.99 if args.gamma is None else float(args.gamma)

    train_env = DummyVecEnv([make_env(args.env_id, seed=args.seed)])
    vec_norm = VecNormalize(
        train_env,
        norm_obs=not args.no_obs_norm,
        norm_reward=not args.no_reward_norm,
        clip_obs=10.0,
        gamma=vn_gamma,
    )

    eval_env = DummyVecEnv([make_env(args.env_id, seed=args.seed + 10)])
    eval_vec_norm = VecNormalize(
        eval_env,
        training=False,
        norm_obs=vec_norm.norm_obs,
        norm_reward=vec_norm.norm_reward,
        clip_obs=10.0,
        gamma=vn_gamma,
    )

    policy_kwargs = (
        dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
        if args.algo == "sac"
        else dict(net_arch=[256, 256])
    )
    model = build_algo(args.algo, vec_norm, seed=args.seed, lr=args.lr,
                       device=args.device, gamma=vn_gamma, policy_kwargs=policy_kwargs)

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.eval_every // 10),
        save_path=str(save_dir / "checkpoints"),
        name_prefix=f"{args.algo}"
    )
    eval_cb = EvalCallback(
        eval_env=eval_vec_norm,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(save_dir / "eval"),
        eval_freq=args.eval_every,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False
    )

    if not args.eval_only:
        print(f"Training {args.algo.upper()} on {args.env_id} for {args.timesteps:,} steps (device={args.device})...")
        model.learn(total_timesteps=args.timesteps, callback=[ckpt_cb, eval_cb], progress_bar=True)

        model_path = save_dir / f"{args.algo}_final"
        model.save(model_path)
        vec_norm.save(save_dir / "vecnormalize.pkl")
        print(f"Saved model to {model_path}.zip and VecNormalize stats to vecnormalize.pkl")

        eval_vec_norm.obs_rms = vec_norm.obs_rms
        eval_vec_norm.ret_rms = vec_norm.ret_rms

    if args.eval_only:
        eval_env2 = DummyVecEnv([make_env(args.env_id, seed=args.seed + 123)])
        stats_path = Path(args.load_path).parent.parent / "vecnormalize.pkl"
        if stats_path.exists():
            eval_vec_norm = VecNormalize.load(str(stats_path), eval_env2)
            eval_vec_norm.training = False
            eval_vec_norm.norm_reward = False
        else:
            eval_vec_norm = VecNormalize(eval_env2, training=False, gamma=vn_gamma)

        Loader = SAC if args.algo == "sac" else PPO
        model = Loader.load(args.load_path, device=args.device)

    mean_r, std_r = evaluate_policy(
        model, eval_vec_norm, n_eval_episodes=args.eval_episodes, deterministic=True, render=False
    )
    print(f"\n=== {args.env_id} ({args.algo.upper()}) returns over {args.eval_episodes} eps ===")
    print(f"Mean: {mean_r:.2f}   Std: {std_r:.2f}")


if __name__ == "__main__":
    main()

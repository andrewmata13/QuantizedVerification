import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from path_utils import get_trained_rl_controller_save_path

def make_env(env_id, seed=0, eval_record_path=None):
    def _thunk():
        if eval_record_path is not None:
            env = gym.make(env_id, render_mode="rgb_array")
            env = RecordVideo(env, video_folder=f"{eval_record_path}/", episode_trigger=lambda episode_id: True)
        else:
            env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk


def build_algo(algo, train_vec_env, seed, lr, device, gamma, policy_kwargs=None):
    algo = algo.lower()
    if algo == "ppo":
        return PPO("MlpPolicy", train_vec_env, verbose=1, seed=seed, learning_rate=lr,
                   policy_kwargs=policy_kwargs, device=device, gamma=gamma)
    if algo == "sac":
        return SAC("MlpPolicy", train_vec_env, verbose=1, seed=seed, learning_rate=lr,
                   policy_kwargs=policy_kwargs, device=device, gamma=gamma)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="HalfCheetah-v4", help="RL Environment Name")
    ap.add_argument("--env_cnt", type=int, default=1, help="Number of environments")
    ap.add_argument("--algo", type=str, default="sac", choices=["ppo", "sac"], help="Choose between PPO and SAC")
    ap.add_argument("--timesteps", type=int, default=3_000_000, help="Total training timesteps")
    ap.add_argument("--seed", type=int, default=0, help="Seed value")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning Rate")
    ap.add_argument("--eval_every", type=int, default=100_000, help="Evaluation interval")
    ap.add_argument("--eval_episodes", type=int, default=10, help="Evaluation Episodes Number")
    ap.add_argument("--gamma", type=float, default=None, help="Discount Factor")
    ap.add_argument("--reward_norm_flag", action="store_true", help="Do we have reward normalization")
    ap.add_argument("--obs_norm_flag", action="store_true", help="Do we have observation normalization")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--eval_only", action="store_true")

    args = ap.parse_args()

    save_path_str, config_str = get_trained_rl_controller_save_path(args)

    save_dir = Path(save_path_str)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "best").mkdir(parents=True, exist_ok=True)
    (save_dir / "eval").mkdir(parents=True, exist_ok=True)

    vn_gamma = 0.99 if args.gamma is None else float(args.gamma)

    train_env = DummyVecEnv([make_env(args.env_id, seed=args.seed + 1000 + i) for i in range(args.env_cnt)])
    train_vec_norm = VecNormalize(
        train_env,
        norm_obs=args.obs_norm_flag,
        norm_reward=args.reward_norm_flag,
        clip_obs=10.0,
        gamma=vn_gamma,
    )

    eval_env = DummyVecEnv([make_env(args.env_id, seed=args.seed + 2000)])
    eval_vec_norm = VecNormalize(
        eval_env,
        training=False,
        norm_obs=args.obs_norm_flag,
        norm_reward=False,
        clip_obs=10.0,
        gamma=vn_gamma,
    )

    policy_kwargs = (
        dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
        if args.algo == "sac"
        else dict(net_arch=[256, 256])
    )
    
    model = build_algo(args.algo, train_vec_norm, seed=args.seed, lr=args.lr,
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

        model_path = save_dir / "model"
        config_path = save_dir / "config.txt"
        model.save(model_path)
        train_vec_norm.save(save_dir / "train_vec_norm.pkl")
        print(f"Saved model to {model_path}.zip and VecNormalize stats to vecnormalize.pkl")

        with open(config_path, "w+") as config_file:
            config_file.write(config_str)

        eval_vec_norm.obs_rms = train_vec_norm.obs_rms
        eval_vec_norm.ret_rms = train_vec_norm.ret_rms

    if args.eval_only:
        load_path = save_path_str
        video_path = f"videos/{save_path_str}"
        video_dir = Path(video_path)
        video_dir.mkdir(parents=True, exist_ok=True)
        config_path = f"{video_path}/config.txt"

        with open(config_path, "w+") as config_file:
            config_file.write(config_str)

        eval_env2 = DummyVecEnv([make_env(args.env_id, seed=args.seed + 3000, eval_record_path=video_path)])
        stats_path = Path(load_path) / "train_vec_norm.pkl"
        if stats_path.exists():
            eval_vec_norm = VecNormalize.load(str(stats_path), eval_env2)
            eval_vec_norm.training = False
            eval_vec_norm.norm_reward = False
        else:
            eval_vec_norm = VecNormalize(eval_env2, training=False, gamma=vn_gamma)

        Loader = SAC if args.algo == "sac" else PPO
        model_path = Path(load_path) / "model.zip"
        if model_path.exists():
            model = Loader.load(model_path, device=args.device)

    mean_r, std_r = evaluate_policy(
        model, eval_vec_norm, n_eval_episodes=args.eval_episodes, deterministic=True, render=True
    )
    print(f"\n=== {args.env_id} ({args.algo.upper()}) returns over {args.eval_episodes} eps ===")
    print(f"Mean: {mean_r:.2f}   Std: {std_r:.2f}")


if __name__ == "__main__":
    main()

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from path_utils import get_trained_rl_controller_save_path, get_dataset_save_path

def save_obj(obs_tensor, action_tensor, name):
    with open(name + "/data.pkl", "wb") as f:
        pickle.dump({"obs": obs_tensor, "action": action_tensor}, f, pickle.HIGHEST_PROTOCOL)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="HalfCheetah-v4", help="RL Environment Name")
    ap.add_argument("--env_cnt", type=int, default=1, help="Number of environments")
    ap.add_argument("--algo", type=str, default="sac", choices=["ppo", "sac"], help="Choose between PPO and SAC")
    ap.add_argument("--n1", type=int, default=[256, 256], nargs="+", help="List of hidden layers of network 1")
    ap.add_argument("--n2", type=int, default=[], nargs="+", help="List of hidden layers of network 2 (For SAC)")
    ap.add_argument("--timesteps", type=int, default=3_000_000, help="Total training timesteps")
    ap.add_argument("--seed", type=int, default=0, help="Seed value")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning Rate")
    ap.add_argument("--eval_every", type=int, default=100_000, help="Evaluation interval")
    ap.add_argument("--eval_episodes", type=int, default=10, help="Evaluation Episodes Number")
    ap.add_argument("--gamma", type=float, default=None, help="Discount Factor")
    ap.add_argument("--reward_norm_flag", action="store_true", help="Do we have reward normalization")
    ap.add_argument("--obs_norm_flag", action="store_true", help="Do we have observation normalization")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--sample_cnt", type=int, default=1000, help="How many samples do you want?")
    ap.add_argument("--discrete_env", action="store_true", help="Is it a discrete environment?")

    args = ap.parse_args()

    load_path, _ = get_trained_rl_controller_save_path(args)

    data_save_path, config_str = get_dataset_save_path(args)

    save_dir = Path(data_save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    eval_env = DummyVecEnv([make_env(args.env_id, seed=2026)])
    stats_path = Path(load_path) / "train_vec_norm.pkl"
    if stats_path.exists():
        eval_vec_norm = VecNormalize.load(str(stats_path), eval_env)
        eval_vec_norm.training = False
        eval_vec_norm.norm_reward = False
    else:
        print("Error in finding environment stats...")

    Loader = SAC if args.algo == "sac" else PPO
    model_path = Path(load_path) / "model.zip"
    if model_path.exists():
        model = Loader.load(model_path, device=args.device)
    
    print("Start generating samples...")

    obs_list = []
    action_list = []

    obs = eval_vec_norm.reset()
    for _ in range(args.sample_cnt):
        action, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs[0])

        if args.discrete_env:
            cur_action = [0 for i in range(eval_vec_norm.action_space.n)]
            cur_action[action[0]] = 1
            action_list.append(cur_action)
        else:
            action_list.append(action.tolist()[0])

        obs, reward, done, info = eval_vec_norm.step(action)
        if done[0]:
            obs = eval_vec_norm.reset()
    
    obs_array = np.stack(obs_list)
    action_array = np.stack(action_list)
    
    obs_tensor = torch.from_numpy(obs_array).float()
    action_tensor = torch.from_numpy(action_array).float()

    save_obj(obs_tensor, action_tensor, data_save_path)
    with open(f"{data_save_path}/config.txt", "w+") as config_file:
        config_file.write(config_str)


if __name__ == "__main__":
    main()
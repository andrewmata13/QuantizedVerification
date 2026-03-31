import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from path_utils import get_dataset_save_path, get_latent_controller_path, get_trained_rl_controller_save_path

# Make sb3 environment
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

# Encoder models
class AffineEncoder(nn.Module):
    def __init__(self, in_dim, m, A=None, b=None):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(m, in_dim))
        self.b = nn.Parameter(torch.zeros(m))
        if A is not None: self.A.data.copy_(A)
        if b is not None: self.b.data.copy_(b)

    def forward(self, x):
        return x @ self.A.t() + self.b

class NonlinearEncoder(nn.Module):
    def __init__(self, in_dim, m, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, m),
        )
        
    def forward(self, x):
        return self.net(x)

# Latent space controller
class LatentController(nn.Module):
    def __init__(self, is_discrete, m, hidden=256, out_dim=6):
        super().__init__()
        self.is_discrete = is_discrete

        if is_discrete:
            self.net = nn.Sequential(
                nn.Linear(m, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, out_dim),
                nn.ReLU()
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(m, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, out_dim),
                nn.Tanh()
            )
    def forward(self, x):
        if self.is_discrete:
            out1 = self.net(x)
            return F.softmax(out1)
        else:
            return self.net(x)

# Collect dataset for learning latent controller
def load_dataset(dataset_path):
    with open(f"{dataset_path}/data.pkl", "rb") as f:
        dataset = pickle.load(f)

    obs = dataset["obs"]
    action = dataset["action"] 
    
    return obs, action  

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
    ap.add_argument("--sample_cnt", type=int, default=1000, help="How many samples do you want?")
    ap.add_argument("--encoder", type=str, default="mlp", choices=["affine","mlp"])
    ap.add_argument("--latent_dim", type=int, default=5)
    ap.add_argument("--latent_hidden", type=int, default=256)
    ap.add_argument("--controller_hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=1024) # batch size
    ap.add_argument("--latent_lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--discrete_env", action="store_true", help="Is it a discrete environment?")


    args = ap.parse_args()
  
    dataset_load_path, _ = get_dataset_save_path(args)
    X, Y = load_dataset(dataset_load_path)
    obs_dim = X.shape[1]
    act_dim = Y.shape[1]

    latent_controller_save_path, config_str = get_latent_controller_path(args)
  
    env_stats_dir, _ = get_trained_rl_controller_save_path(args)

    env_stats_file_path = f"{env_stats_dir}/train_vec_norm.pkl"
    video_path = f"videos/{latent_controller_save_path}"
    video_dir = Path(video_path).mkdir(parents=True, exist_ok=True)
    with open(f"videos/{latent_controller_save_path}/config.txt", "w+") as config_file:
        config_file.write(config_str)


    eval_env = DummyVecEnv([make_env(args.env_id, seed=args.seed + 3000, eval_record_path=video_path)])
    eval_vec_norm = VecNormalize.load(str(env_stats_file_path), eval_env)
    eval_vec_norm.training = False
    eval_vec_norm.norm_reward = False

    # Choose encoder model
    if args.encoder == "affine":
        encoder = AffineEncoder(obs_dim, args.latent_dim)
    else:
        encoder = NonlinearEncoder(obs_dim, args.latent_dim, hidden=args.latent_hidden)

    # Initialize latent controller
    controller = LatentController(args.discrete_env, args.latent_dim, hidden=args.controller_hidden, out_dim=act_dim)

    
    encoder.load_state_dict(torch.load(f"{latent_controller_save_path}/encoder.pth"))
    controller.load_state_dict(torch.load(f"{latent_controller_save_path}/controller.pth"))
    
    encoder.eval()
    controller.eval()

    encoder.to(args.device)
    controller.to(args.device)

    model = lambda x : controller(encoder(torch.from_numpy(x).to(args.device)))

    print("Everything is set...")

    obs = eval_vec_norm.reset()
    tmp = 0

    while tmp < 5:
        action = model(obs).to("cpu").detach().numpy()
        if args.discrete_env:
            action = [np.argmax(action)]
        
        obs, reward, done, info = eval_vec_norm.step(action)
        if done[0]:
            obs = eval_vec_norm.reset()
            tmp += 1


if __name__ == "__main__":
    main()

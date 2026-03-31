import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vnnlib.compat import read_vnnlib_simple

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

import time

from path_utils import get_pinch_controllers_save_path

import nnenum
from nnenum.nnenum import set_control_settings, set_exact_settings, make_spec
from nnenum.settings import Settings
from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.result import Result
from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum.specification import Specification, DisjunctiveSpec
from nnenum.vnnlib import get_num_inputs_outputs, read_vnnlib_simple
from nnenum.lpinstance import SwigArray

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
    def __init__(self, m, hidden=256, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

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
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--spec_id", type=int, default=1, help="Spec id")

    args = ap.parse_args()
  
    pinch_controller_dir, _ = get_pinch_controllers_save_path(args)

    obs_dim = 4
    act_dim = 1
    
    env_stats_file_path = f"{pinch_controller_dir}/train_vec_norm.pkl"
    vnnlib_spec_path = f"specs/{args.env_id}/spec_{args.spec_id}.vnnlib"
    encoder_onnx_path = f"{pinch_controller_dir}/encoder.onnx"
  
    spec_list, input_dtype = make_spec(
        vnnlib_filename=vnnlib_spec_path,
        onnx_filename=encoder_onnx_path,
    )

    network = load_onnx_network_optimized(encoder_onnx_path)

    # Options are "control", "image", "exact"
    set_exact_settings()

    # Set RESULT_SAVE_STARS to True in nnenum settings
    Settings.RESULT_SAVE_STARS = True
    #Settings.TRY_QUICK_OVERAPPROX = True   # default is False for set_exact_settings
    Settings.OVERAPPROX_BOTH_BOUNDS = True  # default is False for set_exact_settings
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX  # default is Settings.BRANCH_EXACT for set_exact_settings

    count = 1
    nnenum_stars = []
    for init_box, spec in spec_list:
        init_box = np.array(init_box, dtype=input_dtype)

        res = enumerate_network(init_box, network, spec)
        result_str = res.result_str
        nnenum_stars += res.stars

    print("*****")


    # Obs Normalization
    # input_spec = np.array(query[0][0]).T

    # output_mat = query[0][1][0][0]
    # output_rhs = query[0][1][0][1]

    # eval_env = DummyVecEnv([make_env(args.env_id, seed=args.seed + 3000)])
    # eval_vec_norm = VecNormalize.load(str(env_stats_file_path), eval_env)
    # eval_vec_norm.training = False
    # eval_vec_norm.norm_reward = False

    # input_spec_normalized = eval_vec_norm.normalize_obs(input_spec)
    # encoder_model = torch.load(encoder_path, weights_only=False)

    exit()




if __name__ == "__main__":
    main()

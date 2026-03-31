import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import load_from_zip_file

from path_utils import get_dataset_save_path, get_latent_controller_path

# Make sb3 environment
def make_env(env_id, seed=0):
    def env_funct():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return env_funct

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


def fit_pca(X, m):
    Xc = X - X.mean(0, keepdim=True)                   # center data
    U,S,Vh = torch.linalg.svd(Xc, full_matrices=False) # find SVD
    A = Vh[:m, :]                                      # take top m principal directions
    b = -(A @ X.mean(0))                               # compute bias
    return A.contiguous(), b.contiguous()       


def train_latent_controller(is_discrete, encoder, controller, X, Y, epochs=20, bs=1024, lr=3e-4, device="cpu"):
    enc, ctrl = encoder.to(device), controller.to(device)
    ds = torch.utils.data.TensorDataset(X, Y)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(list(enc.parameters()) + list(ctrl.parameters()), lr=lr)

    for ep in range(epochs):
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            z  = enc(xb)  
            a  = ctrl(z)  

            # imitation learning
            if is_discrete:
                loss = F.cross_entropy(a,  yb)
            else:
                loss = F.mse_loss(a, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
            
        print(f"[ep {ep+1:02d}] train loss {tot/len(ds):.6f}")

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
    save_dir = Path(latent_controller_save_path)
    (save_dir).mkdir(parents=True, exist_ok=True)  

    # Choose encoder model
    if args.encoder == "affine":
        A, b = fit_pca(X, args.latent_dim)
        encoder = AffineEncoder(obs_dim, args.latent_dim, A=A, b=b)
    else:
        encoder = NonlinearEncoder(obs_dim, args.latent_dim, hidden=args.latent_hidden)

    # Initialize latent controller
    controller = LatentController(args.discrete_env, args.latent_dim, hidden=args.controller_hidden, out_dim=act_dim)

    print("Everything is set...")

    # Train an initial latent controller with the initial grid
    train_latent_controller(
        args.discrete_env, encoder, controller, X, Y,
        epochs=args.epochs, bs=args.bs, lr=args.latent_lr, device=args.device
    )

    torch.save(encoder.state_dict(), f"{latent_controller_save_path}/encoder.pth")
    torch.save(controller.state_dict(), f"{latent_controller_save_path}/controller.pth")
    with open(f"{latent_controller_save_path}/config.txt", "w+") as config_file:
        config_file.write(config_str)



if __name__ == "__main__":
    main()

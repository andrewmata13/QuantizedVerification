import argparse

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import load_from_zip_file


# Load model, bit hacky bc of sb3 model issues
ALGOS = {"PPO": PPO, "SAC": SAC}
def auto_safe_load(path: str, device="auto"):
    data, params, _ = load_from_zip_file(path, device=device, print_system_info=False)

    algo_name = "SAC"
    AlgoCls = ALGOS.get(algo_name)
    if AlgoCls is None:
        raise RuntimeError(f"Unknown/unsupported algo in checkpoint: {algo_name!r}")

    model = AlgoCls.load(path, device=device, custom_objects={"_init_setup_model": False})

    if hasattr(model, "policy_kwargs") and isinstance(model.policy_kwargs, dict):
        for k in list(model.policy_kwargs.keys()):
            if k in {"use_sde", "sde_net_arch", "n_critics"}:
                model.policy_kwargs.pop(k, None)

    if isinstance(model, PPO) and not hasattr(model, "use_sde"):
        model.use_sde = False

    model._setup_model()
    model.set_parameters(params, exact_match=False, device=device)
    return model

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

# Quantization grid on latent space
class AxisQuantizer:
    def __init__(self, mins, maxs, K):
        self.mins = torch.as_tensor(mins, dtype=torch.float32)
        self.maxs = torch.as_tensor(maxs, dtype=torch.float32)
        self.K = K
        self.h = (self.maxs - self.mins) / float(self.K)
        self.device = self.mins.device

    def to(self, device):
        self.mins = self.mins.to(device)
        self.maxs = self.maxs.to(device)
        self.h = self.h.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def snap_center(self, z):
        if z.device != self.device:
            self.to(z.device)
            
        r = (z - self.mins) / self.h
        k = torch.floor(r)
        k = torch.clamp(k, min=0, max=self.K - 1)
        return self.mins + (k + 0.5) * self.h


# Collect dataset for learning latent controller
@torch.no_grad()
def collect_dataset(model, env, steps=200_000):
    obs_list, act_list = [], []
    obs = env.reset()
    t = 0

    # collect observations and actions
    while len(obs_list) < steps:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, rewards, dones, infos = env.step(action)

        obs_list.append(obs.copy().squeeze(0))
        act_list.append(np.clip(action.squeeze(0), -1.0, 1.0))
        obs = next_obs
        t += 1
        if dones[0] or t >= 1000:
            obs = env.reset()
            t = 0
            
    X = torch.tensor(np.array(obs_list), dtype=torch.float32)
    Y = torch.tensor(np.array(act_list), dtype=torch.float32)
    return X, Y


def fit_pca(X, m):
    Xc = X - X.mean(0, keepdim=True)                   # center data
    U,S,Vh = torch.linalg.svd(Xc, full_matrices=False) # find SVD
    A = Vh[:m, :]                                      # take top m principal directions
    b = -(A @ X.mean(0))                               # compute bias
    return A.contiguous(), b.contiguous()


def calibrate_grid(encoder, X, K=12, pad_frac=0.10, device="cpu", K_min=6, K_max=20):
    with torch.no_grad():
        Z = encoder(X.to(device)).cpu()
        
    q_lo = torch.quantile(Z, 0.005, dim=0)
    q_hi = torch.quantile(Z, 0.995, dim=0)
    mins = (q_lo - pad_frac*(q_hi-q_lo)).numpy()
    maxs = (q_hi + pad_frac*(q_hi-q_lo)).numpy()
    K_per_axis = np.full(Z.shape[1], K, dtype=np.int64)
    return mins, maxs, K_per_axis


@torch.no_grad()
def collect_latent_rollout(encoder, controller, env, steps=100_000, device="cpu"):
    zs = []
    obs = env.reset()
    t = 0
    encoder = encoder.to(device).eval()
    controller = controller.to(device).eval()
    while len(zs) < steps:
        ot = torch.tensor(obs, dtype=torch.float32, device=device)
        z  = encoder(ot)
        a  = controller(z).detach().cpu().numpy()
        obs, reward, dones, info = env.step(a)
        zs.append(z.squeeze(0).detach().cpu())
        t += 1
        if dones[0] or t >= 1000:
            obs = env.reset(); t = 0
    return torch.stack(zs, dim=0)             


def train_latent_controller(encoder, controller, quant, X, Y, epochs=20, bs=1024, lr=3e-4, lambda_qat=2e-3, lambda_cons=1e-3, lambda_center=1e-3, device="cpu"):

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
            zc = quant.snap_center(z)
            aq = ctrl(zc)  

            mse   = F.mse_loss(a,  yb)                  # imitation learning
            mseq  = F.mse_loss(aq, yb)                  # quantized imitation
            cons  = F.mse_loss(a,  aq)                  # consistency between paths
            center= torch.mean(((z - zc) / quant.h.to(device))**2)  # pull to center

            loss = mse + lambda_qat*mseq + lambda_cons*cons + lambda_center*center

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
            
        print(f"[ep {ep+1:02d}] train loss {tot/len(ds):.6f}")


@torch.no_grad()
def eval_baseline_sb3(model, env, episodes=10):
    rs = []
    obs = env.reset()
    epi = 0
    steps = 0
    ret = 0
    while epi < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)
        ret += float(reward[0])
        steps += 1
        if dones[0] or steps >= 1000:
            rs.append(ret)
            ret = 0
            steps = 0
            epi += 1
            obs = env.reset()
            
    return float(np.mean(rs)), float(np.std(rs))


@torch.no_grad()
def eval_custom(env, model, episodes=10, device="cpu"):
    rs = []
    obs = env.reset()
    epi = 0
    steps = 0
    ret = 0
    
    while epi < episodes:
        ot = torch.tensor(obs, dtype=torch.float32, device=device)
        action = model(ot).detach().cpu().numpy()
        obs, reward, dones, info = env.step(action)
        ret += reward[0]
        steps += 1
        if dones[0] or steps >= 1000:
            rs.append(ret)
            ret = 0
            steps = 0
            epi += 1
            obs = env.reset()
            
    return float(np.mean(rs)), float(np.std(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="HalfCheetah-v4")
    ap.add_argument("--algo", type=str, default="ppo", choices=["ppo","sac"])
    ap.add_argument("--run_dir", type=str, default="runs/halfcheetah_sb3")
    ap.add_argument("--encoder", type=str, default="mlp", choices=["affine","mlp"])
    ap.add_argument("--latent_dim", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--collect", type=int, default=3_000_000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=1024) # batch size
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--qat", type=float, default=2e-3, help="Quantization-aware training weight.")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    
    run_dir = Path(args.run_dir)
    stats_path = run_dir / "vecnormalize.pkl"

    # Initialize environment
    eval_env = DummyVecEnv([make_env(args.env_id, seed=123)])
    vecnorm = VecNormalize.load(str(stats_path), eval_env)
    vecnorm.training = False
    vecnorm.norm_reward = False

    # Load original model
    model_zip = (run_dir / "best" / "best_model.zip")
    if not model_zip.exists():
        model_zip = run_dir / f"{args.algo}_final.zip"
    Loader = SAC if args.algo == "sac" else PPO
    model = auto_safe_load(str(model_zip), device=args.device)
    
    # Collect dataset for training
    X, Y = collect_dataset(model, vecnorm, steps=args.collect)
    obs_dim = X.shape[1]
    act_dim = Y.shape[1]

    # Choose encoder model
    if args.encoder == "affine":
        A, b = fit_pca(X, args.latent_dim)
        encoder = AffineEncoder(obs_dim, args.latent_dim, A=A, b=b)
    else:
        encoder = NonlinearEncoder(obs_dim, args.latent_dim, hidden=args.hidden)

    # Initialize latent controller
    controller = LatentController(args.latent_dim, hidden=args.hidden, out_dim=act_dim)


    # Create initial grid
    mins, maxs, K_per_axis = calibrate_grid(encoder, X, K=args.K, device="cpu", K_min=6, K_max=20)
    quant = AxisQuantizer(mins, maxs, args.K)

    # Train an initial latent controller with the initial grid
    warmup_epochs = max(3, args.epochs // 5)
    train_latent_controller(
        encoder, controller, quant, X, Y,
        epochs=warmup_epochs, bs=args.bs, lr=args.lr,
        lambda_qat=args.qat, lambda_cons=1e-3, lambda_center=1e-3, device=args.device
    )

    # Create new grid based on trained latent controller
    Z_lat = collect_latent_rollout(encoder, controller, vecnorm,
                                   steps=min(150_000, args.collect // 2), device=args.device)
    q_lo = torch.quantile(Z_lat.cpu(), 0.01, dim=0)
    q_hi = torch.quantile(Z_lat.cpu(), 0.99, dim=0)
    pad_frac = 0.15
    mins2 = (q_lo - pad_frac*(q_hi - q_lo)).numpy()
    maxs2 = (q_hi + pad_frac*(q_hi - q_lo)).numpy()
    K2 = int(max(args.K, 12))
    quant = AxisQuantizer(mins2, maxs2, K2)

    # Finish training latent controller with new grid
    remain_epochs = args.epochs - warmup_epochs
    if remain_epochs > 0:
        train_latent_controller(
            encoder, controller, quant, X, Y,
            epochs=remain_epochs, bs=args.bs, lr=args.lr,
            lambda_qat=args.qat, lambda_cons=1e-3, lambda_center=1e-3, device=args.device
        )

    # Eavluation of baseline, latent, and quantized
    r_base = eval_baseline_sb3(model, vecnorm, episodes=10)
    
    latent_call = lambda ot: controller(encoder(ot.to(args.device)))
    quant_call  = lambda ot: controller(quant.snap_center(encoder(ot.to(args.device))))
    r_lat  = eval_custom(vecnorm, latent_call, episodes=10, device=args.device)
    r_q    = eval_custom(vecnorm,  quant_call, episodes=10, device=args.device)

    print("\nHalfCheetah returns over 10 episodes")
    print(f"Baseline: {r_base[0]:.2f}, {r_base[1]:.2f}")
    print(f"Latent: {r_lat[0]:.2f}, {r_lat[1]:.2f}")
    print(f"Quantized Latent: {r_q[0]:.2f}, {r_q[1]:.2f}")


if __name__ == "__main__":
    main()

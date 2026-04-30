"""
generate_paper_specs.py

Generate 12 VNN-LIB specs per environment for the paper.

Layout (12 specs per env):
  3 easy safe, 3 easy unsafe, 3 hard safe, 3 hard unsafe
  HalfCheetah safe dims: Y_0/Y_1/Y_2, unsafe dims: Y_3/Y_4/Y_5
  Hopper: Y_0/Y_1/Y_2 for both

Box construction:
  1. Collect 50 episodes with the baseline controller
  2. Full bounding box = per-dim [min, max] of all observations
  3. Easy/hard box = centered fraction of full box per dimension

Threshold:
  Safe:   max(PGD(baseline), PGD(bottleneck)) + buffer
  Unsafe: min(PGD(baseline), cell_enum_max(bottleneck)) - buffer
          cell_enum = nnenum encoder stars -> quantized grid -> controller eval

Usage:
    python paper_specs/generate_paper_specs.py
    python paper_specs/generate_paper_specs.py --env HalfCheetah-v4
    python paper_specs/generate_paper_specs.py --easy_frac 0.30 --hard_frac 0.60
"""
import os, sys, argparse

os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(BASE_DIR, "nnenum_package", "nnenum", "src"))
from nnenum.nnenum import set_exact_settings
from nnenum.settings import Settings
Settings.CHECK_SINGLE_THREAD_BLAS = False
from nnenum.enumerate import enumerate_network
from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.specification import Specification
from nnenum.lpinstance import LpInstance

CONFIGS = {
    "HalfCheetah-v4": {
        "baseline_dir": "sac_sweep_runs/HalfCheetah-v4/baseline/seed0",
        "bottleneck_dir": "sac_sweep_runs/HalfCheetah-v4/latent3/seed0",
        "n_actions": 6,
        "quant_step": 0.05,
        "action_dims_safe": [0, 1, 2],
        "action_dims_unsafe": [3, 4, 5],
    },
    "Hopper-v5": {
        "baseline_dir": "sac_sweep_runs/Hopper-v5/baseline512/seed0",
        "bottleneck_dir": "sac_sweep_runs/Hopper-v5/latent3/seed0",
        "n_actions": 3,
        "quant_step": 0.1,
        "action_dims_safe": [0, 1, 2],
        "action_dims_unsafe": [0, 1, 2],
    },
}


def load_networks(cfg, env_name, device="cpu"):
    bdir = os.path.join(BASE_DIR, cfg["baseline_dir"])
    env = DummyVecEnv([lambda: gym.make(env_name)])
    vn = VecNormalize.load(os.path.join(bdir, "train_vec_norm.pkl"), VecMonitor(env))
    vn.training = False
    vn.norm_reward = False
    model = SAC.load(os.path.join(bdir, "model.zip"), env=vn, device=device)
    baseline_net = nn.Sequential(model.actor.latent_pi, model.actor.mu).to(device).eval()

    bndir = os.path.join(BASE_DIR, cfg["bottleneck_dir"])
    enc = torch.load(os.path.join(bndir, "encoder_full.pth"), weights_only=False).to(device).eval()
    ctrl = torch.load(os.path.join(bndir, "latent_controller_full.pth"), weights_only=False).to(device).eval()
    bottleneck_net = nn.Sequential(enc, ctrl).to(device).eval()

    return {"baseline": baseline_net, "bottleneck": bottleneck_net}, model, vn


def collect_episodes(model, vn, n_episodes):
    obs_list = []
    episodes = 0
    obs = vn.reset()
    while episodes < n_episodes:
        act, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs[0].copy())
        obs, _, done, _ = vn.step(act)
        if done[0]:
            episodes += 1
            obs = vn.reset()
    return np.array(obs_list)


def make_box(obs_arr, fraction):
    full_lo = obs_arr.min(axis=0)
    full_hi = obs_arr.max(axis=0)
    center = (full_lo + full_hi) / 2
    half_width = (full_hi - full_lo) * fraction / 2
    return center - half_width, center + half_width


def pgd_max_val(net, lo_t, hi_t, dim, n_restarts=50, n_steps=300,
                batch=512, device="cpu"):
    lo = lo_t.to(device)
    hi = hi_t.to(device)
    best = -1e9
    for _ in range(n_restarts):
        x = lo + torch.rand(batch, lo.shape[0], device=device) * (hi - lo)
        x = x.detach().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=1e-2)
        for _ in range(n_steps):
            opt.zero_grad()
            loss = -net(x)[:, dim].mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.data.clamp_(lo, hi)
        with torch.no_grad():
            val = net(x)[:, dim].max().item()
            best = max(best, val)
    return best


def _cell_reachable(cell, enc_stars, star_bounds, n_latent, quant_step):
    half_q = quant_step / 2.0
    cell_lo = cell - half_q
    cell_hi = cell + half_q
    for s, (s_lo, s_hi) in zip(enc_stars, star_bounds):
        if np.any(s_lo > cell_hi + 1e-9) or np.any(s_hi < cell_lo - 1e-9):
            continue
        if s.a_mat.size == 0:
            if np.all(s.bias >= cell_lo - 1e-9) and np.all(s.bias <= cell_hi + 1e-9):
                return True
            continue
        lpi = LpInstance(s.lpi)
        for d in range(n_latent):
            lpi.add_dense_row(s.a_mat[d], cell_hi[d] - s.bias[d])
            lpi.add_dense_row(-s.a_mat[d], -(cell_lo[d] - s.bias[d]))
        if lpi.minimize(None, fail_on_unsat=False) is not None:
            return True
    return False


def cell_enum_max(encoder_onnx, ctrl_pth, obs_lo, obs_hi, quant_step, device="cpu"):
    """Enumerate reachable quantized cells via nnenum with LP feasibility check,
    evaluate controller, return per-action-dim max."""
    set_exact_settings()
    Settings.RESULT_SAVE_STARS = True
    Settings.OVERAPPROX_BOTH_BOUNDS = True
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX

    net = load_onnx_network_optimized(encoder_onnx)
    n_latent = net.get_num_outputs()
    init_box = np.column_stack([obs_lo, obs_hi]).astype(np.float32)

    trivial_spec = Specification(np.eye(n_latent), np.full(n_latent, -1.0))
    res = enumerate_network(init_box, net, trivial_spec)

    star_bounds = []
    global_lo = np.full(n_latent, np.inf)
    global_hi = np.full(n_latent, -np.inf)
    for s in res.stars:
        s_lo = np.array([s.minimize_output(d, maximize=False) for d in range(n_latent)])
        s_hi = np.array([s.minimize_output(d, maximize=True) for d in range(n_latent)])
        star_bounds.append((s_lo, s_hi))
        global_lo = np.minimum(global_lo, s_lo)
        global_hi = np.maximum(global_hi, s_hi)

    axes = []
    for d in range(n_latent):
        first = np.floor(global_lo[d] / quant_step) * quant_step + quant_step / 2
        last = np.floor(global_hi[d] / quant_step) * quant_step + quant_step / 2
        axes.append(np.arange(first, last + quant_step * 0.5, quant_step))
    grids = np.meshgrid(*axes, indexing="ij")
    all_cells = np.stack([g.ravel() for g in grids], axis=1).astype(np.float32)

    cells = np.array([c for c in all_cells
                       if _cell_reachable(c, res.stars, star_bounds, n_latent, quant_step)],
                      dtype=np.float32)

    if len(cells) == 0:
        print(f"({len(res.stars)} stars, {len(all_cells)} grid, 0 reachable) ",
              end="", flush=True)
        return np.full(net.get_num_outputs(), -np.inf)

    ctrl = torch.load(ctrl_pth, weights_only=False).eval().to(device)
    with torch.no_grad():
        actions = ctrl(torch.tensor(cells, dtype=torch.float32).to(device)).cpu().numpy()

    print(f"({len(res.stars)} stars, {len(all_cells)} grid, {len(cells)} reachable) ",
          end="", flush=True)
    return actions.max(axis=0)


def write_vnnlib(path, obs_lo, obs_hi, n_out, dim, threshold, comment):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_in = len(obs_lo)
    lines = []
    for line in comment.strip().splitlines():
        lines.append(f"; {line}\n")
    lines.append("\n")
    for i in range(n_in):
        lines.append(f"(declare-const X_{i:<2} Real)\n")
    lines.append("\n")
    for i in range(n_out):
        lines.append(f"(declare-const Y_{i:<2} Real)\n")
    lines.append("\n")
    for i in range(n_in):
        lines.append(f"(assert (>= X_{i:<2} {obs_lo[i]:+.4f}))\n")
        lines.append(f"(assert (<= X_{i:<2} {obs_hi[i]:+.4f}))\n")
    lines.append(f"\n(assert (or\n    (and (>= Y_{dim} {threshold:+.6f}))\n))\n")
    with open(path, "w") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, choices=list(CONFIGS.keys()))
    ap.add_argument("--easy_frac", type=float, default=0.10)
    ap.add_argument("--hard_frac", type=float, default=0.20)
    ap.add_argument("--buffer", type=float, default=0.05)
    ap.add_argument("--n_episodes", type=int, default=50)
    args = ap.parse_args()

    envs = [args.env] if args.env else list(CONFIGS.keys())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    for env_name in envs:
        cfg = CONFIGS[env_name]
        spec_dir = os.path.join(BASE_DIR, "paper_specs", env_name)

        print(f"{'='*60}")
        print(f"  {env_name}")
        print(f"{'='*60}")

        print("Loading networks ...")
        nets, model, vn = load_networks(cfg, env_name, device)

        print(f"Collecting {args.n_episodes} episodes ...")
        obs_arr = collect_episodes(model, vn, args.n_episodes)
        vn.close()
        print(f"  {len(obs_arr)} timesteps")

        full_lo = obs_arr.min(axis=0)
        full_hi = obs_arr.max(axis=0)
        print(f"  full box width mean: {(full_hi - full_lo).mean():.4f}")

        boxes = {
            "easy": make_box(obs_arr, args.easy_frac),
            "hard": make_box(obs_arr, args.hard_frac),
        }

        n_actions = cfg["n_actions"]
        spec_num = 1

        bndir = os.path.join(BASE_DIR, cfg["bottleneck_dir"])
        encoder_onnx = os.path.join(bndir, "encoder.onnx")
        ctrl_pth = os.path.join(bndir, "latent_controller_full.pth")
        quant_step = cfg["quant_step"]

        for difficulty, frac in [("easy", args.easy_frac), ("hard", args.hard_frac)]:
            lo, hi = boxes[difficulty]
            lo_t = torch.tensor(lo, dtype=torch.float32)
            hi_t = torch.tensor(hi, dtype=torch.float32)
            print(f"\n  {difficulty.upper()} box ({frac*100:.0f}%)  "
                  f"width mean: {(hi - lo).mean():.4f}")

            cell_max = None

            for safety in ["safe", "unsafe"]:
                dims = cfg["action_dims_safe"] if safety == "safe" else cfg["action_dims_unsafe"]

                if safety == "unsafe" and cell_max is None:
                    print(f"    cell enumeration ... ", end="", flush=True)
                    cell_max = cell_enum_max(
                        encoder_onnx, ctrl_pth, lo, hi, quant_step, device)
                    print(f"per-dim max: {cell_max.round(4).tolist()}")

                for dim in dims:
                    print(f"    spec_{spec_num} ({difficulty}/{safety}/Y_{dim}) ",
                          end="", flush=True)
                    if safety == "safe":
                        pgd_bl = pgd_max_val(nets["baseline"], lo_t, hi_t, dim, device=device)
                        pgd_bn = pgd_max_val(nets["bottleneck"], lo_t, hi_t, dim, device=device)
                        pmax = max(pgd_bl, pgd_bn)
                        buf = abs(pmax) * args.buffer
                        threshold = pmax + buf
                        print(f"PGD bl={pgd_bl:.4f} bn={pgd_bn:.4f}  thr={threshold:.4f}")
                    else:
                        pgd_bl = pgd_max_val(nets["baseline"], lo_t, hi_t, dim, device=device)
                        bn_max = float(cell_max[dim])
                        pmax = min(pgd_bl, bn_max)
                        buf = abs(pmax) * args.buffer
                        threshold = pmax - buf
                        print(f"PGD bl={pgd_bl:.4f} cell bn={bn_max:.4f}  thr={threshold:.4f}")

                    path = os.path.join(spec_dir, f"spec_{spec_num}.vnnlib")
                    write_vnnlib(
                        path, lo, hi, n_actions, dim, threshold,
                        f"{env_name} spec_{spec_num} — {difficulty} {safety}\n"
                        f"Box: {frac*100:.0f}% of full rollout bounding box\n"
                        f"Violation: Y_{dim} >= {threshold:.6f}\n"
                        f"PGD max: {pmax:.4f}, buffer: {args.buffer*100:.0f}%"
                    )
                    spec_num += 1

        print(f"\n  {spec_num - 1} specs written to {spec_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Find a second HalfCheetah unsafe spec where ALL 3 networks violate:
  1. Quantized latent3 (genuinely reachable cells via LP)
  2. Continuous latent3 (PGD)
  3. Baseline continuous (PGD)

For each action dim (0-5), enumerate quantized cell outputs, check LP
reachability of top-violating cells, then PGD-verify continuous networks.
"""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import torch.nn as nn

from nnenum.nnenum import set_exact_settings
from nnenum.settings import Settings
from nnenum.enumerate import enumerate_network
from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.specification import Specification
from nnenum.vnnlib import read_vnnlib_simple, get_num_inputs_outputs
from nnenum.lpinstance import LpInstance

ENV = "HalfCheetah-v4"
RUN_DIR = "sac_sweep_runs/HalfCheetah-v4/latent3/seed0"
BASELINE_DIR = "sac_sweep_runs/HalfCheetah-v4/baseline/seed0"
REF_SPEC = "specs/HalfCheetah-v4/spec_1.vnnlib"
QUANT_STEP = 0.05
N_ACTIONS = 6

EPS_VALUES = [0.01, 0.02, 0.03, 0.05, 0.1]

N_PGD_RESTARTS = 300
PGD_STEPS = 300
PGD_LR = 0.005


class FullNetwork(nn.Module):
    def __init__(self, encoder, controller):
        super().__init__()
        self.encoder = encoder
        self.controller = controller

    def forward(self, x):
        return self.controller(self.encoder(x))


class BaselineNet(nn.Module):
    def __init__(self, latent_pi, mu):
        super().__init__()
        self.latent_pi = latent_pi
        self.mu = mu

    def forward(self, x):
        return self.mu(self.latent_pi(x))


def nnenum_config():
    set_exact_settings()
    Settings.RESULT_SAVE_STARS = True
    Settings.OVERAPPROX_BOTH_BOUNDS = True
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
    Settings.PRINT_OUTPUT = False
    Settings.PRINT_PROGRESS = False


def precompute_star_bounds(enc_stars, n_latent):
    bounds = []
    for s in enc_stars:
        s_lo = np.array([s.minimize_output(d, maximize=False) for d in range(n_latent)])
        s_hi = np.array([s.minimize_output(d, maximize=True) for d in range(n_latent)])
        bounds.append((s_lo, s_hi))
    return bounds


def check_cell_reachable(cell_center, enc_stars, star_bounds, n_latent, quant_step):
    """Check if any encoder star intersects the quantization cell around cell_center."""
    half = quant_step / 2.0
    cell_arr = np.array(cell_center, dtype=float)
    cell_lo = cell_arr - half
    cell_hi = cell_arr + half

    for star, (s_lo, s_hi) in zip(enc_stars, star_bounds):
        if np.any(s_lo > cell_hi + 1e-9) or np.any(s_hi < cell_lo - 1e-9):
            continue

        if star.a_mat.size == 0:
            if np.all(star.bias >= cell_lo - 1e-9) and np.all(star.bias <= cell_hi + 1e-9):
                return True
            continue

        lpi = LpInstance(star.lpi)
        for d in range(n_latent):
            lpi.add_dense_row(star.a_mat[d], cell_hi[d] - star.bias[d])
            lpi.add_dense_row(-star.a_mat[d], -(cell_lo[d] - star.bias[d]))

        if lpi.minimize(None, fail_on_unsat=False) is not None:
            return True
    return False


def pgd_search(model, obs_lo, obs_hi, action_dim, threshold, device,
               n_restarts=N_PGD_RESTARTS, n_steps=PGD_STEPS, lr=PGD_LR):
    lo = torch.tensor(obs_lo, dtype=torch.float32, device=device)
    hi = torch.tensor(obs_hi, dtype=torch.float32, device=device)
    n_in = len(obs_lo)

    best_val = -np.inf
    batch_size = 50

    for restart in range(0, n_restarts, batch_size):
        batch = min(batch_size, n_restarts - restart)
        x = lo + (hi - lo) * torch.rand(batch, n_in, device=device)
        x.requires_grad_(True)

        for step in range(n_steps):
            if x.grad is not None:
                x.grad.zero_()
            out = model(x)
            obj = out[:, action_dim].sum()
            obj.backward()

            with torch.no_grad():
                x = x + lr * x.grad.sign()
                x = torch.clamp(x, lo, hi)
            x = x.detach().requires_grad_(True)

        with torch.no_grad():
            out = model(x)
            vals = out[:, action_dim]
            idx = vals.argmax()
            if vals[idx].item() > best_val:
                best_val = vals[idx].item()

    return best_val


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    encoder_onnx = f"{RUN_DIR}/encoder.onnx"
    latent_ctrl_pth = f"{RUN_DIR}/latent_controller_full.pth"

    n_inputs, n_latent, _ = get_num_inputs_outputs(encoder_onnx)
    print(f"Encoder: {n_inputs} -> {n_latent}")

    # Enumerate reachable cells
    print("\nEnumerating encoder stars...")
    nnenum_config()
    net = load_onnx_network_optimized(encoder_onnx)
    vnnlib_data = read_vnnlib_simple(REF_SPEC, n_inputs, n_latent + N_ACTIONS)
    box, _ = vnnlib_data[0]
    init_box = np.array(box, dtype=np.float32)
    obs_lo = init_box[:, 0]
    obs_hi = init_box[:, 1]

    trivial_spec = Specification(np.eye(n_latent), np.full(n_latent, -1.0))
    res = enumerate_network(init_box, net, trivial_spec)
    enc_stars = res.stars
    print(f"  {len(enc_stars)} encoder stars")

    print("  Precomputing star bounds...")
    star_bounds = precompute_star_bounds(enc_stars, n_latent)

    # Build cell grid from precomputed star bounds
    global_lo = np.full(n_latent, np.inf)
    global_hi = np.full(n_latent, -np.inf)
    for s_lo, s_hi in star_bounds:
        global_lo = np.minimum(global_lo, s_lo)
        global_hi = np.maximum(global_hi, s_hi)

    axes = []
    for d in range(n_latent):
        first = np.floor(global_lo[d] / QUANT_STEP) * QUANT_STEP + QUANT_STEP / 2
        last = np.floor(global_hi[d] / QUANT_STEP) * QUANT_STEP + QUANT_STEP / 2
        axes.append(np.arange(first, last + QUANT_STEP * 0.5, QUANT_STEP))
    grids = np.meshgrid(*axes, indexing="ij")
    cells = np.stack([g.ravel() for g in grids], axis=1).astype(np.float32)
    print(f"  {len(cells):,} total cells in grid")

    # Evaluate latent controller
    ctrl = torch.load(latent_ctrl_pth, weights_only=False).eval().to(device)
    with torch.no_grad():
        z = torch.tensor(cells, dtype=torch.float32, device=device)
        actions = ctrl(z).cpu().numpy()

    # Load continuous models for PGD
    encoder_pth = f"{RUN_DIR}/encoder_full.pth"
    encoder = torch.load(encoder_pth, weights_only=False).eval()
    controller = torch.load(latent_ctrl_pth, weights_only=False).eval()
    latent3_model = FullNetwork(encoder, controller).to(device).eval()

    from stable_baselines3 import SAC
    sb3_model = SAC.load(f"{BASELINE_DIR}/model.zip", device=device)
    actor = sb3_model.actor
    baseline_model = BaselineNet(actor.latent_pi, actor.mu).to(device).eval()

    print("\n" + "="*70)
    print("Scanning all action dims for genuinely reachable unsafe specs")
    print("="*70)

    # Skip dim 0 since spec_9 already uses it
    candidates = []

    for dim in range(N_ACTIONS):
        col = actions[:, dim]
        cell_max = col.max()
        cell_min = col.min()
        print(f"\n--- Action dim {dim}: range [{cell_min:.4f}, {cell_max:.4f}] ---")

        if dim == 0:
            print("  (skipping — already used by spec_9)")
            continue

        # Find top cells and check reachability
        sorted_idx = np.argsort(-col)
        for eps in EPS_VALUES:
            threshold = float(cell_max) - eps
            violating_mask = col > threshold
            n_violating = violating_mask.sum()

            if n_violating == 0:
                continue

            # Check LP reachability of top violating cells (check up to 20)
            violating_indices = np.where(violating_mask)[0]
            n_reachable = 0
            checked = 0
            for idx in violating_indices[:20]:
                cell_center = cells[idx]
                reachable = check_cell_reachable(cell_center, enc_stars, star_bounds, n_latent, QUANT_STEP)
                checked += 1
                if reachable:
                    n_reachable += 1

            print(f"  eps={eps:.3f}: threshold={threshold:.4f}, "
                  f"{n_violating} violating cells, "
                  f"{n_reachable}/{checked} LP-reachable")

            if n_reachable > 0:
                # PGD check on continuous latent3
                lat3_max = pgd_search(latent3_model, obs_lo, obs_hi, dim, threshold, device)
                lat3_unsafe = lat3_max >= threshold

                # PGD check on baseline
                base_max = pgd_search(baseline_model, obs_lo, obs_hi, dim, threshold, device)
                base_unsafe = base_max >= threshold

                print(f"    PGD latent3: max={lat3_max:.4f} {'UNSAFE' if lat3_unsafe else 'safe'}")
                print(f"    PGD baseline: max={base_max:.4f} {'UNSAFE' if base_unsafe else 'safe'}")

                candidates.append({
                    "dim": dim,
                    "eps": eps,
                    "threshold": threshold,
                    "cell_max": float(cell_max),
                    "n_violating": int(n_violating),
                    "n_reachable": n_reachable,
                    "lat3_max": lat3_max,
                    "lat3_unsafe": lat3_unsafe,
                    "base_max": base_max,
                    "base_unsafe": base_unsafe,
                    "all_unsafe": lat3_unsafe and base_unsafe,
                })

                if lat3_unsafe and base_unsafe:
                    print(f"    >>> ALL 3 NETWORKS UNSAFE! dim={dim} eps={eps}")
                    # Found a good one — no need to try more eps values for this dim
                    break

    print("\n" + "="*70)
    print("SUMMARY OF CANDIDATES")
    print("="*70)
    all_three = [c for c in candidates if c["all_unsafe"]]
    if all_three:
        print(f"\n{len(all_three)} candidates where ALL 3 networks are unsafe:")
        for c in all_three:
            print(f"  dim={c['dim']} eps={c['eps']:.3f} threshold={c['threshold']:.4f} "
                  f"({c['n_reachable']} reachable cells) "
                  f"lat3_max={c['lat3_max']:.4f} base_max={c['base_max']:.4f}")
    else:
        print("\nNo candidates found where all 3 networks are unsafe.")
        print("Partial results:")
        for c in candidates:
            status = []
            if c["lat3_unsafe"]: status.append("lat3")
            if c["base_unsafe"]: status.append("base")
            print(f"  dim={c['dim']} eps={c['eps']:.3f} threshold={c['threshold']:.4f} "
                  f"unsafe_for={status or 'none'} "
                  f"lat3_max={c['lat3_max']:.4f} base_max={c['base_max']:.4f}")


if __name__ == "__main__":
    main()

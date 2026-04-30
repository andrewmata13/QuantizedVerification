"""
check_continuous_unsafe.py

For each unsafe spec (9-12), check whether the continuous (non-quantized) full
network also violates the threshold via PGD + random sampling over the full
input box.
"""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import torch.nn as nn

from nnenum.vnnlib import read_vnnlib_simple, get_num_inputs_outputs
from nnenum.specification import Specification, DisjunctiveSpec


CONFIGS = {
    "HalfCheetah-v4": {
        "run_dir":    "sac_sweep_runs/HalfCheetah-v4/latent3/seed0",
        "n_actions":  6,
    },
    "Hopper-v5": {
        "run_dir":    "sac_sweep_runs/Hopper-v5/latent3/seed0",
        "n_actions":  3,
    },
}

N_RANDOM = 1_000_000
N_PGD_RESTARTS = 200
PGD_STEPS = 200
PGD_LR = 0.005


class FullNetwork(nn.Module):
    def __init__(self, encoder, controller):
        super().__init__()
        self.encoder = encoder
        self.controller = controller

    def forward(self, x):
        return self.controller(self.encoder(x))


def parse_violation_dims(action_spec_list):
    """Parse which action dims the spec checks.
    Returns list of (dim, threshold) where violation is Y_dim >= threshold."""
    results = []
    for mat, rhs in action_spec_list:
        mat = np.array(mat, dtype=float)
        rhs = np.array(rhs, dtype=float)
        for i in range(len(rhs)):
            row = mat[i]
            nonzero = np.nonzero(np.abs(row) > 1e-12)[0]
            if len(nonzero) == 1:
                dim = nonzero[0]
                coeff = row[dim]
                # constraint: coeff * Y_dim <= rhs_val
                # if coeff < 0: Y_dim >= rhs_val / coeff
                if coeff < 0:
                    results.append((dim, rhs[i] / coeff))
    return results


def pgd_search(model, obs_lo, obs_hi, action_dim, threshold, device,
               n_restarts=N_PGD_RESTARTS, n_steps=PGD_STEPS, lr=PGD_LR):
    """PGD to maximize action_dim output within the input box."""
    lo = torch.tensor(obs_lo, dtype=torch.float32, device=device)
    hi = torch.tensor(obs_hi, dtype=torch.float32, device=device)
    n_in = len(obs_lo)

    best_val = -np.inf
    best_obs = None
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
                best_obs = x[idx].cpu().numpy()

    return best_obs, best_val, best_val >= threshold


def random_search(model, obs_lo, obs_hi, action_spec_list, device, n_samples=N_RANDOM):
    """Random sampling over the input box through PyTorch model."""
    n_in = len(obs_lo)
    lo = torch.tensor(obs_lo, dtype=torch.float32, device=device)
    hi = torch.tensor(obs_hi, dtype=torch.float32, device=device)

    checker_specs = []
    for mat, rhs in action_spec_list:
        checker_specs.append(Specification(np.array(mat), np.array(rhs)))
    if len(checker_specs) == 1:
        checker = checker_specs[0]
    else:
        checker = DisjunctiveSpec(checker_specs)

    chunk = 100_000
    with torch.no_grad():
        for i in range(0, n_samples, chunk):
            n = min(chunk, n_samples - i)
            samples = lo + (hi - lo) * torch.rand(n, n_in, device=device)
            actions = model(samples).cpu().numpy()

            for j in range(len(actions)):
                if checker.is_violation(actions[j].astype(float)):
                    return samples[j].cpu().numpy(), actions[j], True

    return None, None, False


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 70)
    print("PGD + random search for continuous network violations (specs 9-12)")
    print("=" * 70)

    for env, cfg in CONFIGS.items():
        run_dir = cfg["run_dir"]
        n_actions = cfg["n_actions"]

        encoder_pth = f"{run_dir}/encoder_full.pth"
        ctrl_pth = f"{run_dir}/latent_controller_full.pth"
        full_onnx = f"{run_dir}/full_network.onnx"
        spec_dir = f"specs/{env}"

        n_full_in, n_full_out, _ = get_num_inputs_outputs(full_onnx)

        encoder = torch.load(encoder_pth, weights_only=False).eval()
        controller = torch.load(ctrl_pth, weights_only=False).eval()
        model = FullNetwork(encoder, controller).to(device).eval()

        print(f"\n{'='*60}")
        print(f"  {env} / latent3  (full network {n_full_in}→{n_full_out})")
        print(f"{'='*60}")

        for sid in range(9, 13):
            spec_path = f"{spec_dir}/spec_{sid}.vnnlib"
            if not os.path.exists(spec_path):
                print(f"\n  spec_{sid}: MISSING")
                continue

            vnnlib = read_vnnlib_simple(spec_path, n_full_in, n_full_out)
            box, action_spec_list = vnnlib[0]
            obs_lo = np.array([b[0] for b in box], dtype=np.float32)
            obs_hi = np.array([b[1] for b in box], dtype=np.float32)

            viol_dims = parse_violation_dims(action_spec_list)

            print(f"\n  spec_{sid}:")
            for dim, thr in viol_dims:
                print(f"    Target: Y_{dim} >= {thr:.4f}")

            # PGD search (per violation dimension)
            pgd_found = False
            for dim, thr in viol_dims:
                obs, val, found = pgd_search(
                    model, obs_lo, obs_hi, dim, thr, device)
                print(f"    PGD: Y_{dim} max = {val:.4f}  (threshold {thr:.4f})  "
                      f"{'VIOLATION' if found else 'no violation'}")
                if found:
                    pgd_found = True
                    with torch.no_grad():
                        full_act = model(torch.tensor(obs, device=device).unsqueeze(0))
                    print(f"    PGD witness action: {full_act.cpu().numpy().round(4)}")
                    break

            # Random sampling
            rand_obs, rand_act, rand_found = random_search(
                model, obs_lo, obs_hi, action_spec_list, device)
            if rand_found:
                print(f"    Random: VIOLATION at action = {rand_act.round(4)}")
            else:
                print(f"    Random ({N_RANDOM:,} samples): no violation")

            if pgd_found or rand_found:
                print(f"    >>> CONFIRMED UNSAFE for continuous network")
            else:
                print(f"    >>> No counterexample found (continuous may be safe)")

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()

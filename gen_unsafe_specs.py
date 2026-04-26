"""
gen_unsafe_specs.py

Generate UNSAFE VNN-LIB specs by:
  1. Running our cell enumeration over an existing input box to find the
     true per-action maximum output across all reachable quantized cells.
  2. Setting the threshold just below that maximum (cell_max - epsilon),
     so only a handful of extremal cells constitute a violation.

The resulting specs are provably UNSAFE (we know the counterexample cell),
but the violation is "hidden" — the violating inputs form a thin pre-image
polytope in the high-dimensional observation space, which PGD random restarts
are unlikely to stumble into. This makes α-β CROWN slow even with PGD enabled.

Our method finds the violation immediately by exhaustive cell lookup.

Usage:
    python gen_unsafe_specs.py                        # HalfCheetah latent3
    python gen_unsafe_specs.py --env Hopper-v5 --label latent3
    python gen_unsafe_specs.py --eps 0.05 --n_specs 4
"""

import os, sys, argparse
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch

from nnenum.nnenum import set_exact_settings
from nnenum.settings import Settings
from nnenum.enumerate import enumerate_network
from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.specification import Specification
from nnenum.vnnlib import read_vnnlib_simple, get_num_inputs_outputs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


CONFIGS = {
    "HalfCheetah-v4": {
        "latent3": {
            "run_dir":    "sac_sweep_runs/HalfCheetah-v4/latent3/seed0",
            "quant_step": 0.05,
            "ref_spec":   "specs/HalfCheetah-v4/spec_1.vnnlib",   # borrow input box
            "n_actions":  6,
        },
        "latent2": {
            "run_dir":    "sac_sweep_runs/HalfCheetah-v4/latent2/seed0",
            "quant_step": 0.1,
            "ref_spec":   "specs/HalfCheetah-v4/spec_1.vnnlib",
            "n_actions":  6,
        },
    },
    "Hopper-v5": {
        "latent3": {
            "run_dir":    "sac_sweep_runs/Hopper-v5/latent3/seed0",
            "quant_step": 0.1,
            "ref_spec":   "specs/Hopper-v5/spec_1.vnnlib",
            "n_actions":  3,
        },
        "latent4": {
            "run_dir":    "sac_sweep_runs/Hopper-v5/latent4/seed0",
            "quant_step": 0.1,
            "ref_spec":   "specs/Hopper-v5/spec_1.vnnlib",
            "n_actions":  3,
        },
    },
}


def nnenum_config():
    set_exact_settings()
    Settings.RESULT_SAVE_STARS = True
    Settings.OVERAPPROX_BOTH_BOUNDS = True
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX


def enumerate_cells(encoder_onnx, ref_spec_path, n_inputs, n_latent, quant_step):
    """Run nnenum on encoder and return all reachable quantized cell centers."""
    nnenum_config()
    net = load_onnx_network_optimized(encoder_onnx)
    vnnlib_data = read_vnnlib_simple(ref_spec_path, n_inputs, n_latent + 6)
    box, _ = vnnlib_data[0]
    init_box = np.array(box, dtype=np.float32)

    trivial_spec = Specification(np.eye(n_latent), np.full(n_latent, -1.0))
    res = enumerate_network(init_box, net, trivial_spec)
    enc_stars = res.stars

    global_lo = np.full(n_latent,  np.inf)
    global_hi = np.full(n_latent, -np.inf)
    for s in enc_stars:
        s_lo = np.array([s.minimize_output(d, maximize=False) for d in range(n_latent)])
        s_hi = np.array([s.minimize_output(d, maximize=True)  for d in range(n_latent)])
        global_lo = np.minimum(global_lo, s_lo)
        global_hi = np.maximum(global_hi, s_hi)

    axes = []
    for d in range(n_latent):
        first = np.floor(global_lo[d] / quant_step) * quant_step + quant_step / 2
        last  = np.floor(global_hi[d] / quant_step) * quant_step + quant_step / 2
        axes.append(np.arange(first, last + quant_step * 0.5, quant_step))
    grids = np.meshgrid(*axes, indexing="ij")
    cells = np.stack([g.ravel() for g in grids], axis=1).astype(np.float32)
    return cells, init_box, enc_stars


def eval_cells(cells, latent_ctrl_path):
    """Evaluate latent controller on all cells, return (cells, actions) arrays."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctrl = torch.load(latent_ctrl_path, weights_only=False).eval().to(device)
    chunk = 500_000
    parts = []
    with torch.no_grad():
        for i in range(0, len(cells), chunk):
            z = torch.tensor(cells[i:i+chunk], dtype=torch.float32).to(device)
            parts.append(ctrl(z).cpu().numpy())
    return np.concatenate(parts, axis=0)


def write_vnnlib(path, obs_lo, obs_hi, n_out, violations, comment):
    """violations: list of (op, dim, threshold). op is '>=' or '<='."""
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
    lines.append("\n")
    clauses = [f"    (and ({op} Y_{dim} {thr:+.6f}))" for op, dim, thr in violations]
    lines.append("(assert (or\n" + "\n".join(clauses) + "\n))\n")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env",    default="HalfCheetah-v4",
                    choices=list(CONFIGS.keys()))
    ap.add_argument("--label",  default="latent3")
    ap.add_argument("--eps",    type=float, default=0.05,
                    help="Threshold = cell_max - eps (default 0.05)")
    ap.add_argument("--n_specs", type=int, default=4,
                    help="Number of unsafe specs to generate (one per action dim)")
    args = ap.parse_args()

    cfg = CONFIGS[args.env][args.label]
    run_dir    = cfg["run_dir"]
    quant_step = cfg["quant_step"]
    ref_spec   = cfg["ref_spec"]
    n_actions  = cfg["n_actions"]
    spec_dir   = f"specs/{args.env}"

    encoder_onnx    = f"{run_dir}/encoder.onnx"
    latent_ctrl_pth = f"{run_dir}/latent_controller_full.pth"
    full_onnx       = f"{run_dir}/full_network.onnx"

    from nnenum.vnnlib import get_num_inputs_outputs as _gno
    n_inputs, n_latent, _ = _gno(encoder_onnx)

    print(f"\n{'='*60}")
    print(f"Generating UNSAFE specs: {args.env} / {args.label}")
    print(f"  quant_step={quant_step}  eps={args.eps}  n_specs={args.n_specs}")
    print(f"  encoder: {n_inputs}→{n_latent}  n_actions={n_actions}")
    print(f"{'='*60}")

    print("\nEnumerating reachable cells ...")
    cells, init_box, _ = enumerate_cells(
        encoder_onnx, ref_spec, n_inputs, n_latent, quant_step
    )
    print(f"  {len(cells):,} cells enumerated")

    print("Evaluating latent controller on all cells ...")
    actions = eval_cells(cells, latent_ctrl_pth)  # (N_cells, n_actions)

    obs_lo = init_box[:, 0]
    obs_hi = init_box[:, 1]

    # Per-action dimension: find cell max and the cells that achieve it
    n_per_action = min(args.n_specs, n_actions)
    do_combined = args.n_specs > n_actions  # one extra combined disjunction spec

    per_action_info = []  # (dim, threshold, n_violating, violating_cells)
    for spec_idx in range(n_per_action):
        dim = spec_idx  # action dimension

        col = actions[:, dim]
        cell_max = col.max()
        threshold = float(cell_max) - args.eps

        n_violating = (col > threshold).sum()
        violating_cells = cells[col > threshold]
        per_action_info.append((dim, threshold, n_violating, violating_cells))

        spec_num = 9 + spec_idx
        print(f"\n  spec_{spec_num} (unsafe): Y_{dim} >= {threshold:.6f}")
        print(f"    cell_max = {cell_max:.6f}  ({n_violating} violating cell(s))")
        if n_violating > 0:
            print(f"    example violating cell: {violating_cells[0].tolist()}")

        out_path = f"{spec_dir}/spec_{spec_num}.vnnlib"
        write_vnnlib(
            out_path,
            obs_lo, obs_hi,
            n_actions,
            violations=[(">=", dim, threshold)],
            comment=(
                f"{args.env} UNSAFE spec {spec_idx+1} — {args.label}\n"
                f"Input box: same p20/p80 box as safe spec_1.\n"
                f"Violation: Y_{dim} >= {threshold:.6f}  "
                f"(cell_max={cell_max:.6f}, eps={args.eps})\n"
                f"  {n_violating} violating cell(s) out of {len(cells):,} total.\n"
                f"UNSAFE by construction — counterexample exists at cell "
                f"{violating_cells[0].tolist() if n_violating > 0 else 'N/A'}.\n"
                f"Designed to be hard for PGD: violating inputs form a thin polytope\n"
                f"in the {n_inputs}D observation space."
            ),
        )

    # Combined disjunction spec: violated if ANY action dim exceeds its per-dim threshold
    if do_combined:
        combined_spec_num = 9 + n_per_action
        all_violations = [(">=", dim, thr) for dim, thr, _, _ in per_action_info]
        total_combined = sum(nv for _, _, nv, _ in per_action_info)
        print(f"\n  spec_{combined_spec_num} (unsafe, combined): any Y_dim >= per-dim threshold")
        print(f"    {total_combined} violating cell(s) across all dims")
        out_path = f"{spec_dir}/spec_{combined_spec_num}.vnnlib"
        write_vnnlib(
            out_path,
            obs_lo, obs_hi,
            n_actions,
            violations=all_violations,
            comment=(
                f"{args.env} UNSAFE spec {combined_idx} (combined) — {args.label}\n"
                f"Input box: same p20/p80 box as safe spec_1.\n"
                f"Violation: any Y_dim >= per-dim threshold (disjunction).\n"
                f"  Thresholds: {[(d, f'{t:.4f}') for d, t, _, _ in per_action_info]}\n"
                f"  {total_combined} total violating cells across all dims.\n"
                f"UNSAFE by construction. Harder for PGD: adversary must find\n"
                f"the pre-image of any one of the extremal cells."
            ),
        )
        n_per_action += 1  # count the combined spec in final message

    print(f"\nDone. {n_per_action} unsafe specs written to {spec_dir}/")


if __name__ == "__main__":
    main()

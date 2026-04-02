"""
compare_halfcheetah.py

Verification for HalfCheetah-v4 bottleneck policy using IQ-Verify quantized lookup.

Method: Encoder nnenum + IQ-Verify quantized lookup
    1. Parse input bounds from canonical spec using read_vnnlib_simple.
    2. Build a Specification programmatically (encoder output <= -1, never violated)
       so nnenum enumerates all reachable encoder output star sets.
    3. For each star, use IQ-Verify stateset_to_qpoints (bounding-box method) to
       find quantized latent cell centers — no temp spec file needed.
    4. Evaluate latent_ctrl at each cell center to get pre-tanh actions.
    5. Check action violation conditions parsed from the canonical spec.

Canonical spec format: X_0..X_16 = normalized obs, Y_0..Y_5 = pre-tanh actions.

Usage:
    python compare_halfcheetah.py [--run_dir ...] [--spec_id 4|5|6|7] [--quant_step 0.005]
    python compare_halfcheetah.py --all
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse, time
import numpy as np
import torch

from nnenum.nnenum import set_exact_settings
from nnenum.settings import Settings
from nnenum.enumerate import enumerate_network
from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.specification import Specification, DisjunctiveSpec
from nnenum.vnnlib import read_vnnlib_simple, get_num_inputs_outputs

from iq_verify.set_repr.star_set import StarSet
from iq_verify.quant_reach.quantization_utils import QuantizationUtils

N_INPUTS  = 17
N_ACTIONS = 6


def nnenum_config(overapprox=True):
    set_exact_settings()
    Settings.RESULT_SAVE_STARS = True
    Settings.OVERAPPROX_BOTH_BOUNDS = True
    Settings.BRANCH_MODE = (Settings.BRANCH_OVERAPPROX if overapprox
                            else Settings.BRANCH_EXACT)


def verify(encoder_onnx, spec_path, latent_ctrl_path, quant_step=0.005, overapprox=True):
    """
    Run encoder nnenum + IQ-Verify bounding-box quantized lookup against spec_path.
    Supports any latent dimensionality — auto-detected from encoder ONNX output shape.

    Returns dict with keys: result, t_encoder, t_quant, t_total, n_stars, n_iq_stars,
                             n_cells, n_latent, cells, violations.
    """
    nnenum_config(overapprox)
    encoder_network = load_onnx_network_optimized(encoder_onnx)
    _, n_latent, inp_dtype = get_num_inputs_outputs(encoder_onnx)

    vnnlib_data = read_vnnlib_simple(spec_path, N_INPUTS, N_ACTIONS)
    box, action_spec_list = vnnlib_data[0]
    init_box = np.array(box, dtype=inp_dtype)

    # Trivially unsatisfiable spec for encoder: all outputs <= -1.
    # Encoder outputs are always >= 0 (final ReLU), so this is never violated.
    # nnenum enumerates the full reachable set and returns all output stars.
    trivial_spec = Specification(
        mat=np.eye(n_latent),
        rhs=np.full(n_latent, -1.0)
    )

    # ── Step 1: enumerate encoder output star sets ────────────────────────────
    t0 = time.time()
    res = enumerate_network(init_box, encoder_network, trivial_spec)
    enc_stars = res.stars
    t_encoder = time.time() - t0

    # ── Step 2: convert to IQ-Verify StarSets ────────────────────────────────
    t1 = time.time()
    iq_stars = []
    for s in enc_stars:
        try:
            iq_stars.append(StarSet.from_nnenum_LpStar(s))
        except Exception:
            pass

    # ── Step 3: find quantized cell centers via bounding-box method ───────────
    # For N-dim latent the grid is N-dimensional; stateset_to_qpoints handles this.
    quant_params = [quant_step] * n_latent
    reachable = set()
    for iq_star in iq_stars:
        for pt in QuantizationUtils.stateset_to_qpoints(iq_star, quant_params):
            reachable.add(tuple(round(float(v), 6) for v in pt))
    reachable = sorted(reachable)

    # ── Step 4: evaluate latent_ctrl → pre-tanh actions ──────────────────────
    latent_ctrl = torch.load(latent_ctrl_path, weights_only=False).cpu().eval()
    cells = []
    with torch.no_grad():
        for c in reachable:
            z = torch.tensor([list(c)], dtype=torch.float32)
            action = latent_ctrl(z).squeeze().tolist()
            if not isinstance(action, list):
                action = [action]
            cells.append((c, action))

    # ── Step 5: check action violations ──────────────────────────────────────
    if len(action_spec_list) == 1:
        mat, rhs = action_spec_list[0]
        checker = Specification(mat, rhs)
    else:
        checker = DisjunctiveSpec([Specification(m, r) for m, r in action_spec_list])

    violations = [(c, a) for c, a in cells if checker.is_violation(np.array(a))]
    t_quant = time.time() - t1

    result = "unsafe" if violations else "safe"
    return dict(
        result=result,
        t_encoder=t_encoder,
        t_quant=t_quant,
        t_total=t_encoder + t_quant,
        n_stars=len(enc_stars),
        n_iq_stars=len(iq_stars),
        n_cells=len(reachable),
        n_latent=n_latent,
        cells=cells,
        violations=violations,
    )


def print_result(spec_id, spec_path, r):
    n_act = len(r["cells"][0][1]) if r["cells"] else N_ACTIONS
    print(f"\n{'='*60}")
    print(f"Spec {spec_id}: {spec_path}")
    print(f"{'='*60}")
    print(f"  Result      : {r['result'].upper()}")
    print(f"  Stars       : {r['n_stars']} nnenum  →  {r['n_iq_stars']} IQ-Verify")
    print(f"  Cells       : {r['n_cells']}")
    print(f"  Time        : encoder={r['t_encoder']:.3f}s  quant+eval={r['t_quant']:.3f}s  "
          f"total={r['t_total']:.3f}s")
    if r["violations"]:
        print(f"  Violations  : {len(r['violations'])} cell(s)")

    if r["cells"]:
        a_hdr = " ".join(f"{'a'+str(i):>9}" for i in range(n_act))
        print(f"\n  {'Cell':>9}  {a_hdr}")
        print(f"  {'─'*9}  {'─'*(10*n_act)}")
        violation_set = {c for c, _ in r["violations"]}
        for c, a in r["cells"][:30]:
            a_str = " ".join(f"{v:>+9.4f}" for v in a)
            c_str = "(" + ",".join(f"{v:.3f}" for v in c) + ")" if isinstance(c, tuple) and len(c) > 1 else f"{c[0] if isinstance(c, tuple) else c:>9.4f}"
            marker = "  *** VIOLATION" if c in violation_set else ""
            print(f"  {c_str}  {a_str}{marker}")
        if len(r["cells"]) > 30:
            print(f"  ... ({len(r['cells']) - 30} more cells not shown)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir",    default="sac_sweep_runs/HalfCheetah-v4/arch0/seed0")
    ap.add_argument("--spec_id",    type=int, default=None,
                    help="Single spec to run: 4, 5, 6, or 7.  Omit to run all.")
    ap.add_argument("--all",        action="store_true", help="Run all specs 4-7")
    ap.add_argument("--quant_step", type=float, default=0.005)
    ap.add_argument("--overapprox", action="store_true", default=True)
    args = ap.parse_args()

    encoder_onnx = os.path.join(args.run_dir, "encoder.onnx")
    latent_ctrl  = os.path.join(args.run_dir, "latent_controller_full.pth")

    spec_ids = [4, 5, 6, 7] if (args.all or args.spec_id is None) else [args.spec_id]

    results = {}
    for sid in spec_ids:
        spec_path = f"specs/HalfCheetah-v4/spec_{sid}.vnnlib"
        print(f"\nRunning spec_{sid} ...")
        r = verify(encoder_onnx, spec_path, latent_ctrl,
                   quant_step=args.quant_step, overapprox=args.overapprox)
        results[sid] = r
        print_result(sid, spec_path, r)

    if len(spec_ids) > 1:
        n_lat = next(iter(results.values()))["n_latent"]
        print(f"\n{'='*60}")
        print(f"TIMING SUMMARY  (latent_dim={n_lat})")
        print(f"{'='*60}")
        print(f"  {'Spec':<8}  {'Result':<8}  {'Stars':>6}  {'Cells':>8}  "
              f"{'Encoder(s)':>10}  {'Quant(s)':>9}  {'Total(s)':>9}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*9}  {'─'*9}")
        for sid, r in results.items():
            print(f"  spec_{sid:<3}  {r['result']:<8}  {r['n_stars']:>6}  {r['n_cells']:>8}  "
                  f"{r['t_encoder']:>10.3f}  {r['t_quant']:>9.3f}  {r['t_total']:>9.3f}")


if __name__ == "__main__":
    main()

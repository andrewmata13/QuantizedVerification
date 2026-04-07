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
    python compare_halfcheetah.py [--run_dir ...] [--spec_id 1|2|3|4] [--quant_step 0.005]
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
from nnenum.lpinstance import LpInstance


N_INPUTS  = 17
N_ACTIONS = 6


def nnenum_config(overapprox=True):
    set_exact_settings()
    Settings.RESULT_SAVE_STARS = True
    Settings.OVERAPPROX_BOTH_BOUNDS = True
    Settings.BRANCH_MODE = (Settings.BRANCH_OVERAPPROX if overapprox
                            else Settings.BRANCH_EXACT)


def cell_reachable(cell, enc_stars, star_bounds, n_latent, quant_step):
    """
    Check if a quantized cell center is genuinely reachable by any encoder output star
    via a joint LP feasibility check. Returns True if reachable, False if spurious.

    For each star s that passes a per-dimension bounding box pre-filter, copies s's LP
    and adds constraints: cell_lo[d] <= output[d] <= cell_hi[d] for all d simultaneously.
    If the augmented LP is feasible, the cell is confirmed reachable.
    """
    half_q = quant_step / 2.0
    cell_arr = np.array(cell, dtype=float)
    cell_lo = cell_arr - half_q
    cell_hi = cell_arr + half_q

    for s, (s_lo, s_hi) in zip(enc_stars, star_bounds):
        # Per-dimension bounding box pre-filter (cheap)
        if np.any(s_lo > cell_hi + 1e-9) or np.any(s_hi < cell_lo - 1e-9):
            continue

        # Degenerate star (single point): just check if the point is in the cell
        if s.a_mat.size == 0:
            if np.all(s.bias >= cell_lo - 1e-9) and np.all(s.bias <= cell_hi + 1e-9):
                return True
            continue

        # Joint LP feasibility: copy star's LP and add output box constraints
        lpi = LpInstance(s.lpi)
        for d in range(n_latent):
            # output[d] = a_mat[d] @ alpha + bias[d]
            # upper: a_mat[d] @ alpha <= cell_hi[d] - bias[d]
            lpi.add_dense_row(s.a_mat[d], cell_hi[d] - s.bias[d])
            # lower: -a_mat[d] @ alpha <= -(cell_lo[d] - bias[d])
            lpi.add_dense_row(-s.a_mat[d], -(cell_lo[d] - s.bias[d]))

        if lpi.minimize(None, fail_on_unsat=False) is not None:
            return True

    return False


def verify(encoder_onnx, spec_path, latent_ctrl_path, quant_step=0.005, overapprox=True,
           complete=False):
    """
    Run encoder nnenum + quantized lookup against spec_path.
    Supports any latent dimensionality — auto-detected from encoder ONNX output shape.

    If complete=True, any candidate violations are confirmed via a joint LP feasibility
    check against the encoder output stars, eliminating false positives from the
    bounding-box overapproximation. Safe results are always sound regardless.

    Returns dict with keys: result, t_encoder, t_quant, t_total, n_stars,
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

    # ── Steps 2 & 3: per-star LP output bounds → quantized grid ─────────────
    # Compute global min/max across all stars per latent dimension, then build
    # one grid. Since all per-star grids share the same lattice, their union
    # equals the grid over [global_min, global_max] — no concatenation or
    # deduplication needed (O(S*D) instead of O(N log N)).
    # Per-star bounds are also stored for the optional completeness LP filter.
    t1 = time.time()
    global_lo = np.full(n_latent,  np.inf)
    global_hi = np.full(n_latent, -np.inf)
    star_bounds = []
    for s in enc_stars:
        s_lo = np.array([s.minimize_output(d, maximize=False) for d in range(n_latent)])
        s_hi = np.array([s.minimize_output(d, maximize=True)  for d in range(n_latent)])
        star_bounds.append((s_lo, s_hi))
        global_lo = np.minimum(global_lo, s_lo)
        global_hi = np.maximum(global_hi, s_hi)

    if np.any(global_lo == np.inf):
        reachable = np.zeros((0, n_latent))
    else:
        axes = []
        for d in range(n_latent):
            first = np.floor(global_lo[d] / quant_step) * quant_step + quant_step / 2
            last  = np.floor(global_hi[d] / quant_step) * quant_step + quant_step / 2
            axes.append(np.arange(first, last + quant_step * 0.5, quant_step))
        grids = np.meshgrid(*axes, indexing='ij')
        reachable = np.stack([g.ravel() for g in grids], axis=1).astype(np.float32)

    # ── Step 4: chunked batched GPU forward pass ──────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_ctrl = torch.load(latent_ctrl_path, weights_only=False).to(device).eval()
    cells = []
    if len(reachable):
        chunk_size = 500_000
        actions_parts = []
        with torch.no_grad():
            for i in range(0, len(reachable), chunk_size):
                z_chunk = torch.tensor(reachable[i:i+chunk_size], dtype=torch.float32).to(device)
                actions_parts.append(latent_ctrl(z_chunk).cpu().numpy())
        actions_batch = np.concatenate(actions_parts, axis=0)
        cells = list(zip([tuple(r) for r in reachable], actions_batch.tolist()))

    # ── Step 5: check action violations ──────────────────────────────────────
    if len(action_spec_list) == 1:
        mat, rhs = action_spec_list[0]
        checker = Specification(mat, rhs)
    else:
        checker = DisjunctiveSpec([Specification(m, r) for m, r in action_spec_list])

    violations = [(c, a) for c, a in cells if checker.is_violation(np.array(a, dtype=float))]

    # ── Step 6 (optional): LP completeness filter ─────────────────────────────
    # Confirm each candidate violation is genuinely reachable via joint LP check.
    # Safe results are already sound; only unsafe results may be spurious.
    if complete and violations:
        violations = [(c, a) for c, a in violations
                      if cell_reachable(c, enc_stars, star_bounds, n_latent, quant_step)]

    t_quant = time.time() - t1

    result = "unsafe" if violations else "safe"
    return dict(
        result=result,
        t_encoder=t_encoder,
        t_quant=t_quant,
        t_total=t_encoder + t_quant,
        n_stars=len(enc_stars),
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
    print(f"  Stars       : {r['n_stars']}")
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


ROB_RUN_LABELS = {
    "sac_sweep_runs/HalfCheetah-v4/arch0/seed0":   "latent1",
    "sac_sweep_runs/HalfCheetah-v4/latent2/seed0": "latent2",
    "sac_sweep_runs/HalfCheetah-v4/latent3/seed0": "latent3",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir",    default="sac_sweep_runs/HalfCheetah-v4/arch0/seed0")
    ap.add_argument("--spec_id",    type=int, default=None,
                    help="Single spec to run: 1, 2, 3, or 4.  Omit to run all.")
    ap.add_argument("--all",        action="store_true", help="Run safety specs 1-4")
    ap.add_argument("--rob",        action="store_true",
                    help="Run robustness specs (rob_spec_1..4) for this run_dir")
    ap.add_argument("--rob_all",    action="store_true",
                    help="Run robustness specs for all three latent controllers")
    ap.add_argument("--quant_step", type=float, default=0.005)
    ap.add_argument("--overapprox", action="store_true", default=True)
    ap.add_argument("--complete",   action="store_true", default=False,
                    help="Filter candidate violations via LP membership check (eliminates false positives)")
    args = ap.parse_args()

    encoder_onnx = os.path.join(args.run_dir, "encoder.onnx")
    latent_ctrl  = os.path.join(args.run_dir, "latent_controller_full.pth")

    # ── Robustness spec mode ──────────────────────────────────────────────────
    if args.rob_all:
        run_dirs = list(ROB_RUN_LABELS.keys())
    elif args.rob:
        run_dirs = [args.run_dir]
    else:
        run_dirs = None

    if run_dirs is not None:
        all_results = {}   # (run_label, base_sid) -> result
        for rd in run_dirs:
            label = ROB_RUN_LABELS.get(rd, os.path.basename(os.path.dirname(rd)))
            enc   = os.path.join(rd, "encoder.onnx")
            ctrl  = os.path.join(rd, "latent_controller_full.pth")
            print(f"\n=== {label} ===")
            for base_sid in [1, 2, 3, 4]:
                sp = f"specs/HalfCheetah-v4/rob_spec_{base_sid}_{label}.vnnlib"
                if not os.path.exists(sp):
                    print(f"  Missing {sp} — run gen_robustness_specs.py first")
                    continue
                print(f"  Running rob_spec_{base_sid} ...")
                r = verify(enc, sp, ctrl,
                           quant_step=args.quant_step, overapprox=args.overapprox,
                           complete=args.complete)
                all_results[(label, base_sid)] = r

        # Summary table
        print(f"\n{'='*70}")
        print(f"ROBUSTNESS SUMMARY  (quant_step={args.quant_step})")
        print(f"{'='*70}")
        print(f"  {'Controller':<10}  {'Spec':<8}  {'Result':<8}  {'Stars':>6}  "
              f"{'Cells':>8}  {'Total(s)':>9}")
        print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*9}")
        for (label, sid), r in sorted(all_results.items()):
            print(f"  {label:<10}  rob_{sid:<5}  {r['result']:<8}  {r['n_stars']:>6}  "
                  f"{r['n_cells']:>8}  {r['t_total']:>9.3f}")
        return

    # ── Safety spec mode (original) ───────────────────────────────────────────
    spec_ids = [1, 2, 3, 4] if (args.all or args.spec_id is None) else [args.spec_id]

    results = {}
    for sid in spec_ids:
        spec_path = f"specs/HalfCheetah-v4/spec_{sid}.vnnlib"
        print(f"\nRunning spec_{sid} ...")
        r = verify(encoder_onnx, spec_path, latent_ctrl,
                   quant_step=args.quant_step, overapprox=args.overapprox,
                   complete=args.complete)
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

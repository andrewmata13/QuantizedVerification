"""
verify_policy.py

Unified verification script for any MuJoCo bottleneck policy using the
encoder nnenum + quantized latent lookup method.

Works for any environment (HalfCheetah, Hopper, Walker, Ant, ...) as long as
the run_dir contains:
    encoder.onnx                — obs → latent
    latent_controller_full.pth  — latent → pre-tanh action
    full_network.onnx           — obs → pre-tanh action (used to detect n_actions)

Specs are expected at:
    specs/<env>/spec_<N>.vnnlib           — safety specs
    specs/<env>/rob_spec_<N>_<label>.vnnlib  — robustness specs

Environment / dimensions are auto-detected from the ONNX files — no hardcoding.

Usage:
    # Run safety specs 1-4 for HalfCheetah latent3
    python verify_policy.py --run_dir sac_sweep_runs/HalfCheetah-v4/latent3/seed0 --all

    # Run a single spec
    python verify_policy.py --run_dir sac_sweep_runs/HalfCheetah-v4/latent3/seed0 --spec_id 2

    # Explicit spec path(s)
    python verify_policy.py --run_dir sac_sweep_runs/Hopper-v5/latent3/seed0 \\
        --spec_path specs/Hopper-v5/spec_1.vnnlib

    # Robustness specs for this controller
    python verify_policy.py --run_dir sac_sweep_runs/HalfCheetah-v4/latent3/seed0 --rob

    # Complete mode (LP filter eliminates false-positive violations)
    python verify_policy.py --run_dir sac_sweep_runs/HalfCheetah-v4/latent3/seed0 --all --complete
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


# ── nnenum configuration ──────────────────────────────────────────────────────

def nnenum_config(overapprox=True):
    set_exact_settings()
    Settings.RESULT_SAVE_STARS = True
    Settings.OVERAPPROX_BOTH_BOUNDS = True
    Settings.BRANCH_MODE = (Settings.BRANCH_OVERAPPROX if overapprox
                            else Settings.BRANCH_EXACT)


# ── ONNX dimension helpers ────────────────────────────────────────────────────

def detect_dims(encoder_onnx, full_onnx):
    """Return (n_inputs, n_latent, n_actions, inp_dtype) from ONNX files."""
    n_inputs, n_latent,  inp_dtype = get_num_inputs_outputs(encoder_onnx)
    _,        n_actions, _         = get_num_inputs_outputs(full_onnx)
    return n_inputs, n_latent, n_actions, inp_dtype


# ── LP completeness filter ────────────────────────────────────────────────────

def cell_reachable(cell, enc_stars, star_bounds, n_latent, quant_step):
    """
    Check if a quantized cell center is genuinely reachable by any encoder output star
    via a joint LP feasibility check. Returns True if reachable, False if spurious.

    For each star that passes a per-dimension bounding box pre-filter, copies the star's
    LP and adds cell-box constraints for all dimensions simultaneously.
    """
    half_q = quant_step / 2.0
    cell_arr = np.array(cell, dtype=float)
    cell_lo = cell_arr - half_q
    cell_hi = cell_arr + half_q

    for s, (s_lo, s_hi) in zip(enc_stars, star_bounds):
        if np.any(s_lo > cell_hi + 1e-9) or np.any(s_hi < cell_lo - 1e-9):
            continue

        if s.a_mat.size == 0:
            if np.all(s.bias >= cell_lo - 1e-9) and np.all(s.bias <= cell_hi + 1e-9):
                return True
            continue

        lpi = LpInstance(s.lpi)
        for d in range(n_latent):
            lpi.add_dense_row( s.a_mat[d],  cell_hi[d] - s.bias[d])
            lpi.add_dense_row(-s.a_mat[d], -(cell_lo[d] - s.bias[d]))

        if lpi.minimize(None, fail_on_unsat=False) is not None:
            return True

    return False


# ── Core verification ─────────────────────────────────────────────────────────

def verify(encoder_onnx, spec_path, latent_ctrl_path,
           n_inputs, n_actions,
           quant_step=0.005, overapprox=True, complete=False):
    """
    Run encoder nnenum + quantized lookup against a VNN-LIB spec.

    n_inputs  — number of observation inputs (X_ variables in spec)
    n_actions — number of action outputs (Y_ variables in spec)

    If complete=True, candidate violations are confirmed via joint LP feasibility
    check against encoder output stars, eliminating false positives.

    Returns dict: result, t_encoder, t_quant, t_total, n_stars, n_cells,
                  n_latent, cells, violations.
    """
    nnenum_config(overapprox)
    encoder_network = load_onnx_network_optimized(encoder_onnx)
    _, n_latent, inp_dtype = get_num_inputs_outputs(encoder_onnx)

    vnnlib_data = read_vnnlib_simple(spec_path, n_inputs, n_actions)
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

    # ── Steps 2 & 3: per-star LP output bounds → quantized grid ──────────────
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
                z_chunk = torch.tensor(reachable[i:i+chunk_size],
                                       dtype=torch.float32).to(device)
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
    if complete and violations:
        confirmed = []
        for c, a in violations:
            if cell_reachable(c, enc_stars, star_bounds, n_latent, quant_step):
                confirmed.append((c, a))
                break
        violations = confirmed

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


# ── Output formatting ─────────────────────────────────────────────────────────

def print_result(spec_id, spec_path, r):
    n_act = len(r["cells"][0][1]) if r["cells"] else 0
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

    if r["cells"] and n_act > 0:
        a_hdr = " ".join(f"{'a'+str(i):>9}" for i in range(n_act))
        print(f"\n  {'Cell':>9}  {a_hdr}")
        print(f"  {'─'*9}  {'─'*(10*n_act)}")
        violation_set = {c for c, _ in r["violations"]}
        for c, a in r["cells"][:30]:
            a_str = " ".join(f"{v:>+9.4f}" for v in a)
            c_str = (
                "(" + ",".join(f"{v:.3f}" for v in c) + ")"
                if isinstance(c, tuple) and len(c) > 1
                else f"{c[0] if isinstance(c, tuple) else c:>9.4f}"
            )
            marker = "  *** VIOLATION" if c in violation_set else ""
            print(f"  {c_str}  {a_str}{marker}")
        if len(r["cells"]) > 30:
            print(f"  ... ({len(r['cells']) - 30} more cells not shown)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Encoder nnenum + quantized lookup verification for any MuJoCo env.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--run_dir",    required=True,
                    help="Path to trained model dir containing encoder.onnx, "
                         "full_network.onnx, latent_controller_full.pth")
    ap.add_argument("--env",        default=None,
                    help="Env ID (e.g. HalfCheetah-v4). Auto-derived from run_dir if omitted.")
    ap.add_argument("--spec_id",    type=int, default=None,
                    help="Single safety spec index to run (e.g. 1, 2, 3, 4).")
    ap.add_argument("--spec_path",  type=str, default=None, nargs="+",
                    help="One or more explicit .vnnlib paths to verify.")
    ap.add_argument("--all",        action="store_true",
                    help="Run safety specs 1-4 for this env.")
    ap.add_argument("--rob",        action="store_true",
                    help="Run robustness specs for this run_dir.")
    ap.add_argument("--quant_step", type=float, default=0.005)
    ap.add_argument("--overapprox", action="store_true", default=True)
    ap.add_argument("--complete",   action="store_true", default=False,
                    help="Filter candidate violations via LP membership check.")
    args = ap.parse_args()

    # ── Auto-derive env from run_dir path ─────────────────────────────────────
    # Expected path structure: .../sac_sweep_runs/<env>/<label>/seed<N>
    if args.env is None:
        parts = os.path.normpath(args.run_dir).split(os.sep)
        try:
            idx = parts.index("sac_sweep_runs")
            args.env = parts[idx + 1]
        except (ValueError, IndexError):
            ap.error("Could not auto-derive --env from run_dir path. "
                     "Please pass --env explicitly.")

    run_dir      = args.run_dir
    encoder_onnx = os.path.join(run_dir, "encoder.onnx")
    full_onnx    = os.path.join(run_dir, "full_network.onnx")
    latent_ctrl  = os.path.join(run_dir, "latent_controller_full.pth")
    spec_dir     = os.path.join("specs", args.env)

    for p, name in [(encoder_onnx, "encoder.onnx"),
                    (full_onnx,    "full_network.onnx"),
                    (latent_ctrl,  "latent_controller_full.pth")]:
        if not os.path.exists(p):
            ap.error(f"Required file missing: {p}")

    # ── Detect dims from ONNX ─────────────────────────────────────────────────
    n_inputs, n_latent, n_actions, _ = detect_dims(encoder_onnx, full_onnx)
    label = os.path.basename(os.path.dirname(run_dir))   # e.g. "latent3"
    print(f"  env={args.env}  label={label}  "
          f"n_inputs={n_inputs}  n_latent={n_latent}  n_actions={n_actions}")

    def run_spec(spec_path, spec_id=None):
        sid = spec_id or os.path.basename(spec_path)
        print(f"\nRunning {spec_path} ...")
        r = verify(encoder_onnx, spec_path, latent_ctrl,
                   n_inputs=n_inputs, n_actions=n_actions,
                   quant_step=args.quant_step, overapprox=args.overapprox,
                   complete=args.complete)
        print_result(sid, spec_path, r)
        return r

    # ── Robustness spec mode ──────────────────────────────────────────────────
    if args.rob:
        results = {}
        for sid in range(1, 5):
            sp = os.path.join(spec_dir, f"rob_spec_{sid}_{label}.vnnlib")
            if not os.path.exists(sp):
                print(f"  Missing {sp} — skipping")
                continue
            r = run_spec(sp, f"rob_{sid}")
            results[(label, sid)] = r

        if results:
            print(f"\n{'='*70}")
            print(f"ROBUSTNESS SUMMARY  (quant_step={args.quant_step})")
            print(f"{'='*70}")
            print(f"  {'Controller':<10}  {'Spec':<8}  {'Result':<8}  {'Stars':>6}  "
                  f"{'Cells':>8}  {'Total(s)':>9}")
            print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*9}")
            for (lbl, sid), r in sorted(results.items()):
                print(f"  {lbl:<10}  rob_{sid:<5}  {r['result']:<8}  {r['n_stars']:>6}  "
                      f"{r['n_cells']:>8}  {r['t_total']:>9.3f}")
        return

    # ── Explicit spec path mode ───────────────────────────────────────────────
    if args.spec_path:
        for sp in args.spec_path:
            run_spec(sp)
        return

    # ── Safety spec mode ──────────────────────────────────────────────────────
    spec_ids = [1, 2, 3, 4] if (args.all or args.spec_id is None) else [args.spec_id]

    results = {}
    for sid in spec_ids:
        sp = os.path.join(spec_dir, f"spec_{sid}.vnnlib")
        if not os.path.exists(sp):
            print(f"  Missing {sp} — skipping")
            continue
        results[sid] = run_spec(sp, sid)

    if len(results) > 1:
        n_lat = next(iter(results.values()))["n_latent"]
        print(f"\n{'='*60}")
        print(f"TIMING SUMMARY  (env={args.env}  label={label}  latent_dim={n_lat})")
        print(f"{'='*60}")
        print(f"  {'Spec':<8}  {'Result':<8}  {'Stars':>6}  {'Cells':>8}  "
              f"{'Encoder(s)':>10}  {'Quant(s)':>9}  {'Total(s)':>9}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*9}  {'─'*9}")
        for sid, r in results.items():
            print(f"  spec_{sid:<3}  {r['result']:<8}  {r['n_stars']:>6}  {r['n_cells']:>8}  "
                  f"{r['t_encoder']:>10.3f}  {r['t_quant']:>9.3f}  {r['t_total']:>9.3f}")


if __name__ == "__main__":
    main()

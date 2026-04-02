"""
compare_verification.py

Benchmark three verification strategies for the Hopper-v5 bottleneck policy:

  Method A – Full-network nnenum:
      Run nnenum on the entire network (encoder + latent_ctrl + tanh, 9 layers).
      Verifies action-space properties directly.

  Method B – Encoder-only nnenum:
      Run nnenum on just the encoder (4 layers: 11→16→ReLU→1→ReLU).
      Verifies a latent-space bound for the same input region.
      Much smaller network → faster and produces fewer star splits.

  Method C – Encoder nnenum + Quantized latent lookup:
      1. Run nnenum on the encoder (same as Method B).
      2. Convert nnenum output stars → IQ-Verify StarSet objects.
      3. Quantize the 1-D latent space into cells of width `quant_step`.
      4. Determine which latent cells are reachable from the input region.
      5. For each reachable cell, evaluate latent_ctrl at the cell center
         to get the corresponding deterministic action.
      This gives a compact table: latent_cell → action, which can be used
      for safety checking without running nnenum on the large latent_ctrl.

Usage:
    python compare_verification.py --run_dir sac_sweep_runs/Hopper-v5/arch0/seed0
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse, time, sys
import numpy as np
import torch
import torch.nn as nn

# nnenum
from nnenum.nnenum import set_exact_settings, make_spec
from nnenum.settings import Settings
from nnenum.enumerate import enumerate_network
from nnenum.onnx_network import load_onnx_network_optimized

# iq_verify
from iq_verify.set_repr.star_set import StarSet
from iq_verify.quant_reach.quantization_utils import QuantizationUtils

# ─────────────────────────────────────────────────────────────────────────────
# Shared input spec (spec_3: tight nominal box, 11 inputs)
# Same bounds used for all three methods.
# ─────────────────────────────────────────────────────────────────────────────
INPUT_LO = [0.80, -0.50, -1.0, -1.0, -1.0,  0.5, -1.0, -1.0, -1.0, -1.0, -1.0]
INPUT_HI = [1.20,  0.50,  1.0,  1.0,  1.0,  2.0,  1.0,  1.0,  1.0,  1.0,  1.0]


def build_init_box(lo, hi):
    return np.array(list(zip(lo, hi)), dtype=np.float32)


def nnenum_config(overapprox=True, timeout=None):
    set_exact_settings()
    Settings.RESULT_SAVE_STARS = True
    Settings.OVERAPPROX_BOTH_BOUNDS = True
    Settings.BRANCH_MODE = (Settings.BRANCH_OVERAPPROX if overapprox
                            else Settings.BRANCH_EXACT)
    if timeout is not None:
        Settings.TIMEOUT = timeout


def run_nnenum(onnx_path, spec_path, overapprox=True, timeout=None, label=""):
    """Run nnenum and return (result_str, n_stars, elapsed_sec)."""
    nnenum_config(overapprox, timeout=timeout)
    network = load_onnx_network_optimized(onnx_path)
    spec_list, input_dtype = make_spec(
        vnnlib_filename=spec_path,
        onnx_filename=onnx_path,
    )
    all_stars = []
    t0 = time.time()
    for init_box, spec in spec_list:
        init_box = np.array(init_box, dtype=input_dtype)
        res = enumerate_network(init_box, network, spec)
        all_stars += res.stars
    elapsed = time.time() - t0

    print(f"  [{label}] result={res.result_str}  stars={len(all_stars)}  time={elapsed:.3f}s")
    return res.result_str, all_stars, elapsed


def quantized_latent_analysis(encoder_onnx_path, encoder_spec_path,
                               latent_ctrl_path, quant_step=0.5,
                               overapprox=True):
    """
    Method C: encoder nnenum + quantized latent lookup.

    Returns:
        elapsed_encoder  – time for encoder nnenum
        elapsed_quant    – time for quantized lookup
        reachable_cells  – list of (cell_center, action) pairs
    """
    # ── Step 1: nnenum on encoder ──
    nnenum_config(overapprox)
    network = load_onnx_network_optimized(encoder_onnx_path)
    spec_list, input_dtype = make_spec(
        vnnlib_filename=encoder_spec_path,
        onnx_filename=encoder_onnx_path,
    )
    t0 = time.time()
    enc_stars = []
    for init_box, spec in spec_list:
        init_box = np.array(init_box, dtype=input_dtype)
        res = enumerate_network(init_box, network, spec)
        enc_stars += res.stars
    elapsed_encoder = time.time() - t0
    print(f"  [Encoder nnenum] result={res.result_str}  stars={len(enc_stars)}  time={elapsed_encoder:.3f}s")

    # ── Step 2: convert stars → IQ-Verify StarSets ──
    t1 = time.time()
    iq_stars = []
    for s in enc_stars:
        try:
            iq_star = StarSet.from_nnenum_LpStar(s)
            iq_stars.append(iq_star)
        except Exception as e:
            pass  # skip degenerate stars
    print(f"  [StarSet conversion] {len(enc_stars)} nnenum stars → {len(iq_stars)} IQ-Verify StarSets")

    # ── Step 3: find reachable latent cells ──
    quant_params = [quant_step]  # 1-D quantization
    reachable_qpoints = set()
    for iq_star in iq_stars:
        pts = QuantizationUtils.stateset_to_qpoints(iq_star, quant_params)
        for pt in pts:
            reachable_qpoints.add(round(float(pt[0]), 6))

    reachable_qpoints = sorted(reachable_qpoints)
    print(f"  [Quantization q={quant_step}] {len(reachable_qpoints)} reachable latent cells: "
          f"[{min(reachable_qpoints):.3f}, {max(reachable_qpoints):.3f}]")

    # ── Step 4: evaluate latent_ctrl at each cell center ──
    latent_ctrl = torch.load(latent_ctrl_path, weights_only=False).cpu().eval()
    tanh = nn.Tanh()

    reachable_cells = []
    with torch.no_grad():
        for cell_center in reachable_qpoints:
            z = torch.tensor([[cell_center]], dtype=torch.float32)
            raw_action = latent_ctrl(z)
            action = tanh(raw_action).squeeze().tolist()
            reachable_cells.append((cell_center, action))

    elapsed_quant = time.time() - t1
    return elapsed_encoder, elapsed_quant, reachable_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir",    default="sac_sweep_runs/Hopper-v5/arch0/seed0")
    ap.add_argument("--spec_id",    type=str, default="3",
                    help="Spec ID to verify: 1, 2, 3, 4, 3_calibrated, etc.")
    ap.add_argument("--quant_step", type=float, default=2.0,
                    help="Quantization cell width for the 1-D latent")
    ap.add_argument("--overapprox", action="store_true", default=True,
                    help="Use overapproximation in nnenum (faster)")
    args = ap.parse_args()

    encoder_onnx  = os.path.join(args.run_dir, "encoder.onnx")
    full_onnx     = os.path.join(args.run_dir, "full_network_notanh.onnx")
    latent_ctrl   = os.path.join(args.run_dir, "latent_controller_full.pth")
    encoder_spec  = f"specs/Hopper-v5/spec_{args.spec_id}.vnnlib"
    full_spec     = f"specs/Hopper-v5/spec_{args.spec_id}_full.vnnlib"

    os.makedirs("specs/Hopper-v5", exist_ok=True)
    if not os.path.exists(full_spec):
        _write_full_spec(full_spec)

    FULL_TIMEOUT = 60  # seconds – the 512-dim layers make exact analysis expensive

    print("=" * 60)
    print(f"METHOD A – Full-network nnenum (encoder + latent_ctrl, timeout={FULL_TIMEOUT}s)")
    print("=" * 60)
    _, _, t_full = run_nnenum(full_onnx, full_spec,
                               overapprox=args.overapprox,
                               timeout=FULL_TIMEOUT,
                               label="Full network")

    print()
    print("=" * 60)
    print(f"METHOD B – Encoder nnenum + Quantized lookup (q={args.quant_step})")
    print("=" * 60)
    t_enc_c, t_quant, cells = quantized_latent_analysis(
        encoder_onnx, encoder_spec, latent_ctrl,
        quant_step=args.quant_step,
        overapprox=args.overapprox,
    )

    print()
    print("=" * 60)
    print("RUNTIME SUMMARY")
    print("=" * 60)
    t_enc_c_total = t_enc_c + t_quant
    speedup_c = t_full / t_enc_c_total if t_enc_c_total > 0 else float("inf")
    print(f"  Method A (full nnenum, {FULL_TIMEOUT}s limit) : {t_full:.3f}s")
    print(f"  Method B (encoder + quant lookup) : {t_enc_c_total:.3f}s  ({speedup_c:.1f}× speedup vs A)")
    print(f"    ↳ encoder nnenum: {t_enc_c:.3f}s  quant lookup: {t_quant:.3f}s")

    print()
    print("REACHABLE LATENT CELLS → ACTIONS (Method B)")
    print(f"  {'Cell center':>12}  {'Action [a0, a1, a2]'}")
    print(f"  {'-'*12}  {'-'*40}")
    for cell_center, action in cells:
        a_str = ", ".join(f"{a:+.4f}" for a in action)
        print(f"  {cell_center:>12.3f}  [{a_str}]")


def _write_full_spec(path):
    """Generate full-network spec from the corresponding encoder spec.

    Reads the encoder spec (11 inputs, 1 output), strips everything after
    the last input-bound assert, adds Y_0..Y_2 declarations, and appends
    a pre-tanh bound violation check.
    """
    enc_spec = path.replace("_full.vnnlib", ".vnnlib")
    with open(enc_spec) as f:
        content = f.read()

    # Keep only lines up to and including the last input (X_*) assertion.
    # Drop everything from the first Y_* declaration onward.
    lines = content.splitlines(keepends=True)
    kept = []
    for line in lines:
        if "declare-const Y_" in line or ("assert" in line and "Y_" in line):
            break
        kept.append(line)

    # Strip trailing blank/comment lines then add output section
    while kept and kept[-1].strip() in ("", ";"):
        kept.pop()

    for i in range(3):
        kept.append(f"(declare-const Y_{i}  Real)\n")

    clauses = "\n".join(f"    (and (<= Y_{i} -50.0))" for i in range(3))
    kept.append(f"\n; Violation: any pre-tanh action < -50 (same input box, full network)\n"
                f"(assert (or\n{clauses}\n))\n")

    with open(path, "w") as f:
        f.writelines(kept)
    print(f"  (wrote {path})")


if __name__ == "__main__":
    main()

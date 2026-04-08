# nnenum requires single-threaded BLAS – must be set before numpy/scipy import
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

"""
verify_hopper.py

Verify properties of the Hopper-v5 bottleneck encoder using nnenum.

Pipeline:
  1. Load encoder.onnx from a trained run directory
  2. Load VNN-LIB spec(s) from specs/Hopper-v5/
  3. Run nnenum exact reachability analysis
  4. Report UNSAT (property proved) / SAT (counterexample found)
  5. Optionally collect the star sets and compute quantized reachability

Usage:
    python verify_hopper.py --run_dir sac_sweep_runs/Hopper-v5/arch0/seed0 --spec_id 1
    python verify_hopper.py --run_dir sac_sweep_runs/Hopper-v5/arch0/seed0 --spec_id 2
    python verify_hopper.py --run_dir sac_sweep_runs/Hopper-v5/arch0/seed0 --spec_id 3_calibrated
"""
import os, argparse, time
import numpy as np

import nnenum
from nnenum.nnenum import set_exact_settings, make_spec
from nnenum.settings import Settings
from nnenum.enumerate import enumerate_network
from nnenum.onnx_network import load_onnx_network_optimized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str,
                    default="sac_sweep_runs/Hopper-v5/arch0/seed0",
                    help="Run directory produced by train_custom_sb3.py")
    ap.add_argument("--spec_id", type=str, default="1",
                    help="Spec identifier: 1, 2, 3, 3_calibrated, 4_forward_motion")
    ap.add_argument("--overapprox", action="store_true",
                    help="Use faster overapproximation instead of exact analysis")
    args = ap.parse_args()

    encoder_onnx_path = os.path.join(args.run_dir, "encoder.onnx")
    spec_path = f"specs/Hopper-v5/spec_{args.spec_id}.vnnlib"

    if not os.path.exists(encoder_onnx_path):
        raise FileNotFoundError(
            f"encoder.onnx not found at {encoder_onnx_path}.\n"
            "Run train_custom_sb3.py first."
        )
    if not os.path.exists(spec_path):
        raise FileNotFoundError(
            f"Spec not found at {spec_path}.\n"
            "For calibrated specs run generate_hopper_specs.py first."
        )

    print(f"Encoder : {encoder_onnx_path}")
    print(f"Spec    : {spec_path}")
    print()

    # Load network
    network = load_onnx_network_optimized(encoder_onnx_path)
    print(f"Loaded encoder: {network}")

    # Build spec
    spec_list, input_dtype = make_spec(
        vnnlib_filename=spec_path,
        onnx_filename=encoder_onnx_path,
    )

    # Configure nnenum
    set_exact_settings()
    Settings.RESULT_SAVE_STARS = True
    Settings.OVERAPPROX_BOTH_BOUNDS = True

    if args.overapprox:
        Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
        print("Mode: overapproximation (fast)")
    else:
        Settings.BRANCH_MODE = Settings.BRANCH_EXACT
        print("Mode: exact")

    all_stars = []
    t0 = time.time()

    for i, (init_box, spec) in enumerate(spec_list):
        init_box = np.array(init_box, dtype=input_dtype)
        print(f"\nVerifying sub-spec {i+1}/{len(spec_list)}…")
        res = enumerate_network(init_box, network, spec)
        all_stars += res.stars

        elapsed = time.time() - t0
        print(f"  Result  : {res.result_str}")
        print(f"  Stars   : {len(res.stars)}")
        print(f"  Elapsed : {elapsed:.2f}s")

        if res.result_str == "unsafe-unsat":
            print("  → UNSAT: property PROVED for this input region.")
        elif res.result_str.startswith("safe"):
            print("  → SAT: property VIOLATED – counterexample found.")
            if hasattr(res, 'cinput') and res.cinput is not None:
                print(f"    Counterexample input  : {res.cinput}")
                print(f"    Counterexample output : {res.coutput}")
        else:
            print(f"  → {res.result_str}")

    elapsed_total = time.time() - t0
    print(f"\n=== Total: {len(all_stars)} stars in {elapsed_total:.2f}s ===")

    # Report latent output range from stars
    if all_stars:
        try:
            from nnenum.lpinstance import LpInstance
            latent_mins = []
            latent_maxs = []
            for star in all_stars:
                # Each star covers a linear region; get the latent bounds
                lp = star.lpi
                # Minimize Y_0
                lo = lp.minimize(lp.get_num_cols() - 1)
                # Maximize Y_0 = minimize -Y_0
                hi = -lp.minimize(-(lp.get_num_cols() - 1))
                latent_mins.append(lo)
                latent_maxs.append(hi)
            print(f"\nVerified latent output range: [{min(latent_mins):.4f}, {max(latent_maxs):.4f}]")
            print("(Union of all star sets from nnenum exact analysis)")
        except Exception as e:
            print(f"\n(Could not compute latent range from stars: {e})")


if __name__ == "__main__":
    main()

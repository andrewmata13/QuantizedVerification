"""
compare_abcrown.py

Run alpha-beta CROWN on all specs for all networks and compare timing/results
against our quantized-bottleneck method.

For bottleneck networks: verifies the full concatenated network (encoder +
latent_controller, no Tanh) — i.e. without the quantized lookup trick.
For baseline [256,256]: only alpha-beta CROWN is run (our method doesn't apply).

Requires:
  - alpha-beta CROWN installed: pip install -e /path/to/alpha-beta-CROWN
  - Full network ONNX files: run export_full_onnx.py first

Usage:
    python compare_abcrown.py                            # all envs, all specs
    python compare_abcrown.py --env HalfCheetah-v4
    python compare_abcrown.py --env HalfCheetah-v4 --timeout 120
    python compare_abcrown.py --spec_type safety         # safety specs only
    python compare_abcrown.py --spec_type traj           # trajectory specs only
    python compare_abcrown.py --input_split              # use input splitting
"""

import os, time, argparse
import torch

# ── Network configs ────────────────────────────────────────────────────────────

NETWORKS = {
    "HalfCheetah-v4": [
        {
            "label":   "baseline",
            "run_dir": "sac_sweep_runs/HalfCheetah-v4/baseline/seed0",
            "is_baseline": True,
        },
        {
            "label":   "latent1",
            "run_dir": "sac_sweep_runs/HalfCheetah-v4/arch0/seed0",
            "quant_step": 0.02,
            "is_baseline": False,
        },
        {
            "label":   "latent2",
            "run_dir": "sac_sweep_runs/HalfCheetah-v4/latent2/seed0",
            "quant_step": 0.1,
            "is_baseline": False,
        },
        {
            "label":   "latent3",
            "run_dir": "sac_sweep_runs/HalfCheetah-v4/latent3/seed0",
            "quant_step": 0.05,
            "is_baseline": False,
        },
    ],
    "Hopper-v5": [
        {
            "label":   "baseline",
            "run_dir": "sac_sweep_runs/Hopper-v5/baseline/seed0",
            "is_baseline": True,
        },
        {
            "label":   "latent2",
            "run_dir": "sac_sweep_runs/Hopper-v5/latent2/seed0",
            "quant_step": None,   # fill after training
            "is_baseline": False,
        },
        {
            "label":   "latent3",
            "run_dir": "sac_sweep_runs/Hopper-v5/latent3/seed0",
            "quant_step": None,
            "is_baseline": False,
        },
        {
            "label":   "latent4",
            "run_dir": "sac_sweep_runs/Hopper-v5/latent4/seed0",
            "quant_step": None,
            "is_baseline": False,
        },
    ],
}

# Results from our quantized method (verified on new p20/p80 nominal-running specs)
# All 4 specs: nominal running box, checking Y_4, Y_3, Y_5, Y_1 upper bounds
OUR_RESULTS = {
    # (env, label, spec_name): (result, total_seconds)
    ("HalfCheetah-v4", "latent1", "spec_1"): ("safe", 1.600),
    ("HalfCheetah-v4", "latent1", "spec_2"): ("safe", 1.480),
    ("HalfCheetah-v4", "latent1", "spec_3"): ("safe", 1.490),
    ("HalfCheetah-v4", "latent1", "spec_4"): ("safe", 1.310),
    ("HalfCheetah-v4", "latent2", "spec_1"): ("safe", 1.440),
    ("HalfCheetah-v4", "latent2", "spec_2"): ("safe", 1.530),
    ("HalfCheetah-v4", "latent2", "spec_3"): ("safe", 1.350),
    ("HalfCheetah-v4", "latent2", "spec_4"): ("safe", 1.400),
    ("HalfCheetah-v4", "latent3", "spec_1"): ("safe", 1.640),
    ("HalfCheetah-v4", "latent3", "spec_2"): ("safe", 1.680),
    ("HalfCheetah-v4", "latent3", "spec_3"): ("safe", 1.570),
    ("HalfCheetah-v4", "latent3", "spec_4"): ("safe", 1.800),
}


# ── Spec discovery ─────────────────────────────────────────────────────────────

def find_specs(env, spec_type, label):
    spec_dir = f"specs/{env}"
    if not os.path.isdir(spec_dir):
        return []

    specs = []
    if spec_type in ("safety", "all"):
        for i in range(1, 5):
            p = os.path.join(spec_dir, f"spec_{i}.vnnlib")
            if os.path.exists(p):
                specs.append((f"spec_{i}", p))

    if spec_type in ("traj", "all"):
        for i in range(1, 6):
            for kind in ("unsafe", "safe"):
                p = os.path.join(spec_dir, f"traj_spec_{i}_{label}_{kind}.vnnlib")
                if os.path.exists(p):
                    specs.append((f"traj_{i}_{kind}", p))

    return specs


# ── alpha-beta CROWN runner ────────────────────────────────────────────────────

# Status strings from abcrown that mean "property holds" (safe/unsat)
_SAFE_STATUSES   = {"verified", "safe", "safe-incomplete"}
# Status strings that mean "violation found" (unsafe/sat)
_UNSAFE_STATUSES = {"unsafe-pgd", "unsafe-bab", "falsified"}

# Lazily cached — set once on first call to run_abcrown
_abcrown_imports = {}


def _get_abcrown():
    """Import abcrown API classes, applying device config first."""
    if not _abcrown_imports:
        from abcrown import ABCrownSolver, VerificationSpec, ConfigBuilder
        _abcrown_imports.update(
            ABCrownSolver=ABCrownSolver,
            VerificationSpec=VerificationSpec,
            ConfigBuilder=ConfigBuilder,
        )
    return _abcrown_imports


def _apply_config_globally(cfg_dict):
    """Push a config dict into abcrown's global arguments.Config.

    VerificationSpec.build_spec reads arguments.Config['general']['device']
    during vnnlib parsing, so the config must be applied before building specs.
    """
    import arguments
    from api import _deep_update, _clone_config, _ensure_config_defaults
    _ensure_config_defaults()
    new_cfg = _clone_config(arguments.Config.all_args)
    _deep_update(new_cfg, cfg_dict)
    arguments.Config.all_args = new_cfg
    arguments.Config.update_arguments()


def run_abcrown(onnx_path, vnnlib_path, timeout, batch_size=1024,
                input_split=False):
    """Run alpha-beta CROWN via Python API. Returns (result_str, elapsed_seconds)."""
    api = _get_abcrown()
    ABCrownSolver = api["ABCrownSolver"]
    VerificationSpec = api["VerificationSpec"]
    ConfigBuilder = api["ConfigBuilder"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    builder = (
        ConfigBuilder.from_defaults()
        .set(general__device=device)
        .set(solver__batch_size=batch_size)
        .set(bab__timeout=timeout)
        .set(attack__pgd_order="skip")   # skip PGD pre-attack for fair timing
    )
    if input_split:
        builder.set(bab__branching__input_split__enable=True)
    cfg = builder()

    # Apply config globally before building spec (device must be set for vnnlib parsing)
    _apply_config_globally(cfg)

    spec = VerificationSpec.build_spec(vnnlib_path=os.path.abspath(vnnlib_path))
    solver = ABCrownSolver(spec, os.path.abspath(onnx_path), config=cfg)
    t0 = time.time()
    result = solver.solve()
    elapsed = time.time() - t0

    status = str(getattr(result, "status", "unknown"))
    if status in _SAFE_STATUSES:
        label = "safe"
    elif status in _UNSAFE_STATUSES:
        label = "unsafe"
    elif "timeout" in status.lower() or "unknown" in status.lower():
        label = "timeout"
    else:
        label = status

    return label, elapsed


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env",       default=None, choices=list(NETWORKS.keys()),
                    help="Environment to run. Default: all.")
    ap.add_argument("--spec_type", default="safety", choices=["safety", "traj", "all"])
    ap.add_argument("--timeout",   type=int, default=300,
                    help="Per-spec timeout in seconds for alpha-beta CROWN.")
    ap.add_argument("--batch_size", type=int, default=1024,
                    help="Solver batch size (reduce if OOM).")
    ap.add_argument("--labels",    nargs="*", default=None,
                    help="Limit to specific controller labels.")
    ap.add_argument("--input_split", action="store_true",
                    help="Use input splitting instead of neuron splitting.")
    args = ap.parse_args()

    envs = [args.env] if args.env else list(NETWORKS.keys())

    split_tag = " [input-split]" if args.input_split else ""
    all_rows = []

    for env in envs:
        for net in NETWORKS[env]:
            label = net["label"]
            if args.labels and label not in args.labels:
                continue

            onnx_path = os.path.join(net["run_dir"], "full_network.onnx")
            if not os.path.exists(onnx_path):
                print(f"  SKIP {env}/{label} — {onnx_path} missing (run export_full_onnx.py)")
                continue

            specs = find_specs(env, args.spec_type, label)
            if not specs:
                print(f"  SKIP {env}/{label} — no specs found")
                continue

            print(f"\n=== {env} / {label}{split_tag} ===")
            for spec_name, spec_path in specs:
                print(f"  {spec_name} ... ", end="", flush=True)
                abc_result, abc_time = run_abcrown(
                    onnx_path, spec_path, args.timeout, args.batch_size,
                    input_split=args.input_split,
                )
                our = OUR_RESULTS.get((env, label, spec_name))
                our_result = our[0] if our else "N/A"
                our_time   = f"{our[1]:.2f}s" if our else "N/A"
                speedup    = f"{abc_time/our[1]:.1f}×" if our and abc_time > 0 else "N/A"
                print(f"{abc_result:<10} {abc_time:6.1f}s  "
                      f"(ours: {our_result} {our_time}, speedup {speedup})")
                all_rows.append((env, label, spec_name, abc_result, abc_time,
                                 our_result, our_time, speedup))

    # Summary table
    if all_rows:
        print(f"\n{'='*90}")
        print(f"COMPARISON SUMMARY  (timeout={args.timeout}s{split_tag})")
        print(f"{'='*90}")
        print(f"  {'Env':<16} {'Controller':<10} {'Spec':<12} "
              f"{'α-β-CROWN':>10} {'Time(s)':>8}  {'Ours':>8} {'OurTime':>8}  {'Speedup':>8}")
        print(f"  {'─'*16} {'─'*10} {'─'*12} {'─'*10} {'─'*8}  {'─'*8} {'─'*8}  {'─'*8}")
        for env, label, spec, abc_r, abc_t, our_r, our_t, spdup in all_rows:
            print(f"  {env:<16} {label:<10} {spec:<12} "
                  f"{abc_r:>10} {abc_t:>8.1f}  {our_r:>8} {our_t:>8}  {spdup:>8}")


if __name__ == "__main__":
    main()

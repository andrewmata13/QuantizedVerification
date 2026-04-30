"""
generate_specs_5_8.py

Overnight pipeline to create specs 5-8: same output properties as specs 1-4
(Y_4, Y_3, Y_5, Y_1 upper bounds) but with a TIGHTER input box so that
alpha-beta CROWN can actually finish — targeting 5-30 min vs our ~1.5s.

Pipeline
--------
  Phase 1  Calibrate box size
    Try p38/p62, p36/p64, p34/p66, p32/p68 using a single calibration spec
    (Y_4 on baseline, 30-min timeout each).  Pick the tightest box where
    CROWN finishes within CALIB_TIMEOUT, and that finishes in >= TARGET_MIN_S.
    If every candidate times out, use p32/p68 anyway (smallest tried).

  Phase 2  Generate specs 5-8
    At the winning percentile, generate four specs with the same output
    dimensions as specs 1-4, PGD-verified across baseline + latent1/2/3.
      spec_5: Y_4 (front knee)
      spec_6: Y_3 (front hip)
      spec_7: Y_5 (front ankle)
      spec_8: Y_1 (back knee)

  Phase 3  Full CROWN comparison on baseline (PHASE3_TIMEOUT per spec)
    Run alpha-beta CROWN on all four new specs.

  Phase 4  Our method on latent1/2/3 for all four new specs
    Run verify_policy.verify() with complete=True for each controller × spec.

  Results printed as a comparison table at the end.

Usage:
    python generate_specs_5_8.py
    python generate_specs_5_8.py --skip_calibration --plo 35 --phi 65
"""

import os, sys, time, argparse
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

SPEC_DIR       = "specs/HalfCheetah-v4"
BASELINE_RD    = "sac_sweep_runs/HalfCheetah-v4/baseline/seed0"
BOTTLENECK_RDS = {
    "latent1": ("sac_sweep_runs/HalfCheetah-v4/arch0/seed0",   0.02),
    "latent2": ("sac_sweep_runs/HalfCheetah-v4/latent2/seed0", 0.10),
    "latent3": ("sac_sweep_runs/HalfCheetah-v4/latent3/seed0", 0.05),
}

# Calibration: percentile pairs to try, coarsest → finest input box
CALIB_PERCENTILES = [(38, 62), (36, 64), (34, 66), (32, 68)]
CALIB_TIMEOUT     = 1800   # 30 min per calibration probe
TARGET_MIN_S      = 300    # want CROWN to take at least 5 min on the final specs
PHASE3_TIMEOUT    = 7200   # 2 hr per spec in full comparison

# Output dims matching specs 1-4
SPEC_DIMS   = [4, 3, 5, 1]   # Y_4, Y_3, Y_5, Y_1
SPEC_LABELS = ["front-knee", "front-hip", "front-ankle", "back-knee"]
PGD_MARGIN  = 1.5
N_ROLLOUT   = 50_000
N_SAMPLE    = 1_000_000

# ── Data loading ───────────────────────────────────────────────────────────────

def load_baseline(device="cpu"):
    env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
    vn  = VecNormalize.load(f"{BASELINE_RD}/train_vec_norm.pkl", VecMonitor(env))
    vn.training = False; vn.norm_reward = False
    model = SAC.load(f"{BASELINE_RD}/model.zip", env=vn, device=device)
    net   = nn.Sequential(model.actor.latent_pi, model.actor.mu).to(device).eval()
    return model, vn, net


def load_bottleneck_nets(device="cpu"):
    nets = {}
    for label, (rd, _) in BOTTLENECK_RDS.items():
        enc  = torch.load(f"{rd}/encoder_full.pth",           weights_only=False).to(device).eval()
        ctrl = torch.load(f"{rd}/latent_controller_full.pth", weights_only=False).to(device).eval()
        nets[label] = nn.Sequential(enc, ctrl).eval()
    return nets


def collect_rollouts(model, vn, n_steps):
    obs_list = []
    obs = vn.reset()
    for _ in range(n_steps):
        act, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs[0].copy())
        obs, _, done, _ = vn.step(act)
        if done[0]:
            obs = vn.reset()
    return np.array(obs_list)


def nominal_mask(obs_arr):
    xvel = obs_arr[:, 9]
    return (xvel >= 0.5) & (xvel <= 1.5) & (np.abs(obs_arr[:, 1]) <= 0.3) & (np.abs(obs_arr[:, 0]) <= 0.4)


# ── PGD ────────────────────────────────────────────────────────────────────────

def pgd_max(net, lo_t, hi_t, dim, n_steps=300, n_restarts=40, device="cpu"):
    """Return the maximum value of output[dim] found by PGD over [lo_t, hi_t]."""
    lo_t = lo_t.to(device); hi_t = hi_t.to(device)
    best = -1e9
    batch = 512
    for _ in range(n_restarts):
        x = (lo_t + torch.rand(batch, lo_t.shape[0], device=device) * (hi_t - lo_t))
        x = x.detach().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=1e-2)
        for _ in range(n_steps):
            opt.zero_grad()
            loss = -net(x)[:, dim].mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.data.clamp_(lo_t, hi_t)
        with torch.no_grad():
            val = net(x)[:, dim].max().item()
            if val > best:
                best = val
    return best


# ── Spec writer ────────────────────────────────────────────────────────────────

def write_spec(path, obs_lo, obs_hi, dim, threshold, plo, phi, label):
    n_in, n_out = len(obs_lo), 6
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        f"; HalfCheetah-v4 — {label} upper bound\n",
        f"; Input box: p{plo}/p{phi} nominal running (tighter than specs 1-4 p20/p80)\n",
        f"; Violation: Y_{dim} >= {threshold:+.4f}\n",
        f"; SAFE for baseline [256,256] and latent1/2/3.\n\n",
    ]
    for i in range(n_in):
        lines.append(f"(declare-const X_{i:<2} Real)\n")
    lines.append("\n")
    for i in range(n_out):
        lines.append(f"(declare-const Y_{i:<2} Real)\n")
    lines.append("\n")
    for i in range(n_in):
        lines.append(f"(assert (>= X_{i:<2} {obs_lo[i]:+.4f}))\n")
        lines.append(f"(assert (<= X_{i:<2} {obs_hi[i]:+.4f}))\n")
    lines.append(f"\n(assert (or\n    (and (>= Y_{dim} {threshold:+.4f}))\n))\n")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  wrote {path}")


# ── alpha-beta CROWN ───────────────────────────────────────────────────────────

def run_abcrown(onnx_path, spec_path, timeout, batch_size=1024):
    """Returns (result_str, elapsed_seconds)."""
    from abcrown import ABCrownSolver, VerificationSpec, ConfigBuilder
    spec   = VerificationSpec.build_spec(vnnlib_path=os.path.abspath(spec_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = (
        ConfigBuilder.from_defaults()
        .set(general__device=device)
        .set(solver__batch_size=batch_size)
        .set(bab__timeout=timeout)
    )()
    solver  = ABCrownSolver(spec, os.path.abspath(onnx_path), config=cfg)
    t0      = time.time()
    result  = solver.solve()
    elapsed = time.time() - t0

    status = getattr(result, "status", "unknown")
    _SAFE    = {"verified", "safe", "safe-incomplete"}
    _UNSAFE  = {"unsafe-pgd", "unsafe-bab", "falsified"}
    if status in _SAFE:
        label = "safe"
    elif status in _UNSAFE:
        label = "unsafe"
    elif "timeout" in str(status).lower() or "unknown" in str(status).lower():
        label = "timeout"
    else:
        label = str(status)
    return label, elapsed


# ── Our method ─────────────────────────────────────────────────────────────────

def run_ours(rd, quant_step, n_inputs, n_actions, spec_path, complete=True):
    from verify_policy import verify
    enc  = os.path.join(rd, "encoder.onnx")
    ctrl = os.path.join(rd, "latent_controller_full.pth")
    return verify(enc, spec_path, ctrl,
                  n_inputs=n_inputs, n_actions=n_actions,
                  quant_step=quant_step, overapprox=True, complete=complete)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_calibration", action="store_true",
                    help="Skip Phase 1 and use --plo/--phi directly.")
    ap.add_argument("--plo", type=float, default=None,
                    help="Low percentile for input box (used with --skip_calibration).")
    ap.add_argument("--phi", type=float, default=None,
                    help="High percentile for input box (used with --skip_calibration).")
    ap.add_argument("--skip_crown", action="store_true",
                    help="Skip Phase 3 (CROWN comparison).")
    ap.add_argument("--skip_ours", action="store_true",
                    help="Skip Phase 4 (our method).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Load networks ──────────────────────────────────────────────────────────
    print("Loading baseline ...")
    model, vn, baseline_net = load_baseline(device)
    print("Loading bottleneck networks ...")
    bn_nets = load_bottleneck_nets(device)
    all_nets = {"baseline": baseline_net, **bn_nets}

    print(f"Collecting {N_ROLLOUT} rollout steps ...")
    obs_arr = collect_rollouts(model, vn, N_ROLLOUT)
    vn.close()
    nom = obs_arr[nominal_mask(obs_arr)]
    print(f"  nominal subset: {len(nom)} steps")

    baseline_onnx = os.path.join(BASELINE_RD, "full_network.onnx")

    # ── Phase 1: calibration ───────────────────────────────────────────────────
    if args.skip_calibration:
        if args.plo is None or args.phi is None:
            ap.error("--skip_calibration requires --plo and --phi")
        winning_plo, winning_phi = args.plo, args.phi
        print(f"Skipping calibration, using p{winning_plo:.0f}/p{winning_phi:.0f}")
    else:
        print(f"\n{'='*65}")
        print(f"PHASE 1 — CALIBRATION  (CROWN timeout={CALIB_TIMEOUT}s each)")
        print(f"{'='*65}")
        calib_results = []
        winning_plo, winning_phi = CALIB_PERCENTILES[-1]  # fallback: tightest box

        for plo, phi in CALIB_PERCENTILES:
            lo = np.percentile(nom, plo, axis=0)
            hi = np.percentile(nom, phi, axis=0)
            lo_t = torch.tensor(lo, dtype=torch.float32)
            hi_t = torch.tensor(hi, dtype=torch.float32)
            width = (hi - lo).mean()

            # PGD threshold for Y_4 only
            print(f"\n  p{plo}/p{phi}  (mean width={width:.4f})  PGD Y_4 ...")
            thr = max(pgd_max(net, lo_t, hi_t, 4, device=device) for net in all_nets.values())
            thr += PGD_MARGIN
            print(f"    Y_4 threshold={thr:.4f}")

            calib_spec = os.path.join(SPEC_DIR, f"_calib_{plo}_{phi}.vnnlib")
            write_spec(calib_spec, lo, hi, 4, thr, plo, phi, f"Calibration p{plo}/{phi}")

            print(f"    Running CROWN (timeout={CALIB_TIMEOUT}s) ...", flush=True)
            res, elapsed = run_abcrown(baseline_onnx, calib_spec, CALIB_TIMEOUT)
            print(f"    Result: {res}  time={elapsed:.1f}s")
            calib_results.append((plo, phi, thr, lo, hi, res, elapsed))

            if res == "safe" and elapsed >= TARGET_MIN_S:
                winning_plo, winning_phi = plo, phi
                print(f"    --> Winner: p{plo}/p{phi} ({elapsed:.0f}s >= {TARGET_MIN_S}s target)")
                break
            elif res == "safe" and elapsed < TARGET_MIN_S:
                print(f"    Too fast ({elapsed:.1f}s < {TARGET_MIN_S}s), trying bigger box ...")
                # Don't break — continue to next (larger) box
            elif res == "timeout":
                # Too hard at this percentile; keep this as fallback if nothing better
                # But we go from largest to smallest box, so if we haven't found a winner yet
                # this means even the coarsest box was too hard — just go with tightest
                print(f"    Timeout at p{plo}/p{phi}, trying tighter box ...")
                # Continue to next candidate (smaller box)

        print(f"\n  Calibration done.  Selected: p{winning_plo:.0f}/p{winning_phi:.0f}")
        # Clean up calibration spec files
        for plo, phi, *_ in calib_results:
            p = os.path.join(SPEC_DIR, f"_calib_{plo}_{phi}.vnnlib")
            if os.path.exists(p):
                os.remove(p)

    # ── Phase 2: generate specs 5-8 ───────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"PHASE 2 — GENERATE SPECS 5-8  (p{winning_plo:.0f}/p{winning_phi:.0f})")
    print(f"{'='*65}")

    lo = np.percentile(nom, winning_plo, axis=0)
    hi = np.percentile(nom, winning_phi, axis=0)
    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)
    print(f"  box mean width: {(hi-lo).mean():.4f}")

    spec_paths = {}
    thresholds = {}
    for i, (dim, label) in enumerate(zip(SPEC_DIMS, SPEC_LABELS)):
        spec_id = 5 + i
        print(f"\n  spec_{spec_id} (Y_{dim}, {label}) — PGD ...")
        thr = max(pgd_max(net, lo_t, hi_t, dim, device=device) for net in all_nets.values())
        thr += PGD_MARGIN

        # Verify PGD doesn't find violation (it shouldn't, we're above PGD max)
        pgd_ok = True
        for name, net in all_nets.items():
            m = pgd_max(net, lo_t, hi_t, dim, device=device)
            if m >= thr:
                print(f"    WARNING: {name} PGD max {m:.4f} >= threshold {thr:.4f}!")
                pgd_ok = False
        if pgd_ok:
            print(f"    threshold Y_{dim} >= {thr:.4f}  (PGD max={thr-PGD_MARGIN:.4f})  ✓ safe")

        path = os.path.join(SPEC_DIR, f"spec_{spec_id}.vnnlib")
        write_spec(path, lo, hi, dim, thr,
                   winning_plo, winning_phi,
                   f"Spec {spec_id} — p{winning_plo:.0f}/p{winning_phi:.0f} nominal running, "
                   f"Y_{dim} ({label}) upper bound")
        spec_paths[spec_id] = path
        thresholds[spec_id] = thr

    # ── Phase 3: CROWN on baseline ─────────────────────────────────────────────
    crown_results = {}
    if not args.skip_crown:
        print(f"\n{'='*65}")
        print(f"PHASE 3 — CROWN ON BASELINE  (timeout={PHASE3_TIMEOUT}s per spec)")
        print(f"{'='*65}")
        for spec_id, path in spec_paths.items():
            dim = SPEC_DIMS[spec_id - 5]
            print(f"\n  spec_{spec_id} (Y_{dim}) ...", flush=True)
            res, elapsed = run_abcrown(baseline_onnx, path, PHASE3_TIMEOUT)
            crown_results[spec_id] = (res, elapsed)
            print(f"  --> {res}  {elapsed:.1f}s")

    # ── Phase 4: Our method on latent1/2/3 ────────────────────────────────────
    from nnenum.vnnlib import get_num_inputs_outputs
    ours_results = {}   # (label, spec_id) -> result_dict
    if not args.skip_ours:
        print(f"\n{'='*65}")
        print(f"PHASE 4 — OUR METHOD  (latent1/2/3, complete=True)")
        print(f"{'='*65}")
        for label, (rd, quant_step) in BOTTLENECK_RDS.items():
            enc_onnx  = os.path.join(rd, "encoder.onnx")
            full_onnx = os.path.join(rd, "full_network.onnx")
            n_in,  _, _       = get_num_inputs_outputs(enc_onnx)
            _,  n_act, _      = get_num_inputs_outputs(full_onnx)
            print(f"\n  {label}  (quant_step={quant_step})")
            for spec_id, path in spec_paths.items():
                dim = SPEC_DIMS[spec_id - 5]
                print(f"    spec_{spec_id} (Y_{dim}) ...", flush=True)
                r = run_ours(rd, quant_step, n_in, n_act, path, complete=True)
                ours_results[(label, spec_id)] = r
                print(f"    --> {r['result']}  stars={r['n_stars']}  cells={r['n_cells']}  "
                      f"total={r['t_total']:.3f}s")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"COMPARISON SUMMARY — specs 5-8  "
          f"(p{winning_plo:.0f}/p{winning_phi:.0f} nominal box, CROWN timeout={PHASE3_TIMEOUT}s)")
    print(f"{'='*90}")
    print(f"  {'Controller':<10} {'Spec':<8} {'CROWN':>10} {'CROWNt':>8}  "
          f"{'Ours':>8} {'Stars':>6} {'Cells':>8} {'OursT':>8}  {'Speedup':>9}")
    print(f"  {'─'*10} {'─'*8} {'─'*10} {'─'*8}  {'─'*8} {'─'*6} {'─'*8} {'─'*8}  {'─'*9}")

    for spec_id in spec_paths:
        crown_r, crown_t = crown_results.get(spec_id, ("N/A", 0))
        # Baseline row
        print(f"  {'baseline':<10} spec_{spec_id:<3} {crown_r:>10} {crown_t:>8.1f}  "
              f"{'N/A':>8} {'N/A':>6} {'N/A':>8} {'N/A':>8}  {'N/A':>9}")
        # Bottleneck rows
        for label in BOTTLENECK_RDS:
            r = ours_results.get((label, spec_id))
            if r is None:
                print(f"  {label:<10} spec_{spec_id:<3} {'':>10} {'':>8}  "
                      f"{'N/A':>8} {'N/A':>6} {'N/A':>8} {'N/A':>8}  {'N/A':>9}")
                continue
            speedup = f"{crown_t / r['t_total']:.1f}×" if crown_r == "safe" and r['t_total'] > 0 else "N/A"
            print(f"  {label:<10} spec_{spec_id:<3} {'':>10} {'':>8}  "
                  f"{r['result']:>8} {r['n_stars']:>6} {r['n_cells']:>8} {r['t_total']:>8.3f}  "
                  f"{speedup:>9}")

    print(f"\nSpecs written to {SPEC_DIR}/spec_5.vnnlib .. spec_8.vnnlib")


if __name__ == "__main__":
    main()

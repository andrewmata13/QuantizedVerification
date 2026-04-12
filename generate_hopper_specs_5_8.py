"""
generate_hopper_specs_5_8.py

Generate Hopper-v5 specs 5-8: same output properties as specs 1-4
(Y_0 thigh, Y_1 leg, Y_2 foot upper bounds + any-action OR clause)
but with a TIGHTER input box so that alpha-beta CROWN can finish in
a meaningful time (targeting ~5-30 min vs our <40s), enabling a fair
timing comparison rather than a pure timeout.

Mirrors generate_specs_5_8.py (HalfCheetah) in structure.

Pipeline
--------
  Phase 1  Calibrate box size
    Try p38/p62, p36/p64, p34/p66, p32/p68 on the baseline full network
    (calibration spec: Y_0, CALIB_TIMEOUT each).
    Pick the tightest box where CROWN finishes within CALIB_TIMEOUT
    and takes at least TARGET_MIN_S.  Fallback: p32/p68.

  Phase 2  Generate specs 5-8
    spec_5: Y_0 (thigh joint)
    spec_6: Y_1 (leg joint)
    spec_7: Y_2 (foot joint)
    spec_8: any Y_i >= threshold (OR clause, mirrors spec_4)

  Phase 3  CROWN on baseline full network (PHASE3_TIMEOUT per spec)

  Phase 4  Our method on latent3 / latent4 (quant_step=0.1, complete=True)

Usage:
    python generate_hopper_specs_5_8.py
    python generate_hopper_specs_5_8.py --skip_calibration --plo 35 --phi 65
    python generate_hopper_specs_5_8.py --skip_crown   # skip CROWN, gen specs + run ours
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

SPEC_DIR    = "specs/Hopper-v5"
BASELINE_RD = "sac_sweep_runs/Hopper-v5/baseline/seed0"
BOTTLENECK_RDS = {
    "latent3": ("sac_sweep_runs/Hopper-v5/latent3/seed0", 0.1),
    "latent4": ("sac_sweep_runs/Hopper-v5/latent4/seed0", 0.1),
}

N_OBS     = 11
N_ACT     = 3
N_ROLLOUT = 50_000

# Calibration: percentile pairs to try, coarsest → finest input box
CALIB_PERCENTILES = [(38, 62), (36, 64), (34, 66), (32, 68)]
CALIB_TIMEOUT     = 1800   # 30 min per calibration probe
TARGET_MIN_S      = 300    # want CROWN to take at least 5 min on final specs
PHASE3_TIMEOUT    = 7200   # 2 hr per spec in full CROWN comparison
PGD_MARGIN        = 1.5

# Output dims matching specs 1-4
SPEC_DIMS   = [0, 1, 2]          # Y_0 thigh, Y_1 leg, Y_2 foot  (specs 5-7)
SPEC_LABELS = ["thigh", "leg", "foot"]
# spec_8 is the OR-clause over all dims (mirrors spec_4)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_baseline(device="cpu"):
    env = DummyVecEnv([lambda: gym.make("Hopper-v5")])
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
    """Nominal balanced hopping: forward velocity in [0.5,1.5], height/pitch bounded."""
    xvel   = obs_arr[:, 5]
    height = obs_arr[:, 0]
    pitch  = obs_arr[:, 1]
    return (
        (xvel >= 0.5) & (xvel <= 1.5) &
        (np.abs(height) <= 1.5) &
        (np.abs(pitch)  <= 1.0)
    )


# ── PGD ────────────────────────────────────────────────────────────────────────

def pgd_max(net, lo_t, hi_t, dim, n_steps=500, n_restarts=40, device="cpu"):
    """Return the maximum value of output[dim] found by PGD over [lo_t, hi_t]."""
    lo_t = lo_t.to(device); hi_t = hi_t.to(device)
    best = -1e9
    for _ in range(n_restarts):
        x = (lo_t + torch.rand(256, N_OBS, device=device) * (hi_t - lo_t))
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


# ── Spec writers ───────────────────────────────────────────────────────────────

def write_spec_single(path, obs_lo, obs_hi, dim, threshold, plo, phi, label):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        f"; Hopper-v5 — {label}\n",
        f"; Input box: p{plo}/p{phi} nominal hopping "
        f"(tighter than specs 1-4 p20/p80)\n",
        f"; Violation: Y_{dim} >= {threshold:+.4f}\n",
        f"; SAFE for baseline [256,256] and latent3/latent4.\n\n",
    ]
    for i in range(N_OBS):
        lines.append(f"(declare-const X_{i:<2} Real)\n")
    lines.append("\n")
    for i in range(N_ACT):
        lines.append(f"(declare-const Y_{i:<2} Real)\n")
    lines.append("\n")
    for i in range(N_OBS):
        lines.append(f"(assert (>= X_{i:<2} {obs_lo[i]:+.4f}))\n")
        lines.append(f"(assert (<= X_{i:<2} {obs_hi[i]:+.4f}))\n")
    lines.append(f"\n(assert (or\n    (and (>= Y_{dim} {threshold:+.4f}))\n))\n")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  wrote {path}")


def write_spec_or(path, obs_lo, obs_hi, threshold, plo, phi):
    """OR-clause over all N_ACT output dims (mirrors spec_4)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        f"; Hopper-v5 — spec_8 — any-action upper saturation bound\n",
        f"; Input box: p{plo}/p{phi} nominal hopping "
        f"(tighter than specs 1-4 p20/p80)\n",
        f"; Violation: any Y_i >= {threshold:+.4f}\n",
        f"; SAFE for baseline [256,256] and latent3/latent4.\n\n",
    ]
    for i in range(N_OBS):
        lines.append(f"(declare-const X_{i:<2} Real)\n")
    lines.append("\n")
    for i in range(N_ACT):
        lines.append(f"(declare-const Y_{i:<2} Real)\n")
    lines.append("\n")
    for i in range(N_OBS):
        lines.append(f"(assert (>= X_{i:<2} {obs_lo[i]:+.4f}))\n")
        lines.append(f"(assert (<= X_{i:<2} {obs_hi[i]:+.4f}))\n")
    clauses = [f"    (and (>= Y_{i} {threshold:+.4f}))" for i in range(N_ACT)]
    lines.append("(assert (or\n" + "\n".join(clauses) + "\n))\n")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  wrote {path}")


# ── alpha-beta CROWN ───────────────────────────────────────────────────────────

def run_abcrown(onnx_path, spec_path, timeout, batch_size=1024):
    """Returns (result_str, elapsed_seconds)."""
    from compare_abcrown import _apply_config_globally
    from abcrown import ABCrownSolver, VerificationSpec, ConfigBuilder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = (
        ConfigBuilder.from_defaults()
        .set(general__device=device)
        .set(solver__batch_size=batch_size)
        .set(bab__timeout=timeout)
        .set(attack__pgd_order="skip")
    )()
    _apply_config_globally(cfg)

    spec   = VerificationSpec.build_spec(vnnlib_path=os.path.abspath(spec_path))
    solver = ABCrownSolver(spec, os.path.abspath(onnx_path), config=cfg)
    t0     = time.time()
    result = solver.solve()
    elapsed = time.time() - t0

    status = str(getattr(result, "status", "unknown"))
    _SAFE   = {"verified", "safe", "safe-incomplete"}
    _UNSAFE = {"unsafe-pgd", "unsafe-bab", "falsified"}
    if status in _SAFE:
        label = "safe"
    elif status in _UNSAFE:
        label = "unsafe"
    elif "timeout" in status.lower() or "unknown" in status.lower():
        label = "timeout"
    else:
        label = status
    return label, elapsed


# ── Our method ─────────────────────────────────────────────────────────────────

def run_ours(rd, quant_step, spec_path, complete=True):
    from verify_policy import verify
    from nnenum.vnnlib import get_num_inputs_outputs
    enc_onnx  = os.path.join(rd, "encoder.onnx")
    full_onnx = os.path.join(rd, "full_network.onnx")
    n_in,  _, _ = get_num_inputs_outputs(enc_onnx)
    _, n_act, _ = get_num_inputs_outputs(full_onnx)
    ctrl_path   = os.path.join(rd, "latent_controller_full.pth")
    return verify(enc_onnx, spec_path, ctrl_path,
                  n_inputs=n_in, n_actions=n_act,
                  quant_step=quant_step, overapprox=True, complete=complete)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_calibration", action="store_true",
                    help="Skip Phase 1 and use --plo/--phi directly.")
    ap.add_argument("--plo", type=float, default=None)
    ap.add_argument("--phi", type=float, default=None)
    ap.add_argument("--skip_crown", action="store_true",
                    help="Skip Phase 3 (CROWN on baseline).")
    ap.add_argument("--skip_ours", action="store_true",
                    help="Skip Phase 4 (our method on latent3/latent4).")
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
    print(f"  nominal subset: {len(nom)} steps ({100*len(nom)/len(obs_arr):.1f}%)")

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

            print(f"\n  p{plo}/p{phi}  (mean width={width:.4f})  PGD Y_0 ...")
            thr = max(pgd_max(net, lo_t, hi_t, 0, device=device)
                      for net in all_nets.values()) + PGD_MARGIN
            print(f"    Y_0 threshold={thr:.4f}")

            calib_spec = os.path.join(SPEC_DIR, f"_calib_{plo}_{phi}.vnnlib")
            write_spec_single(calib_spec, lo, hi, 0, thr, plo, phi,
                              f"Calibration p{plo}/{phi} Y_0")

            print(f"    Running CROWN (timeout={CALIB_TIMEOUT}s) ...", flush=True)
            res, elapsed = run_abcrown(baseline_onnx, calib_spec, CALIB_TIMEOUT)
            print(f"    Result: {res}  time={elapsed:.1f}s")
            calib_results.append((plo, phi, res, elapsed))

            if res == "safe" and elapsed >= TARGET_MIN_S:
                winning_plo, winning_phi = plo, phi
                print(f"    --> Winner: p{plo}/p{phi} ({elapsed:.0f}s >= {TARGET_MIN_S}s)")
                break
            elif res == "safe":
                print(f"    Too fast ({elapsed:.1f}s < {TARGET_MIN_S}s), trying bigger box ...")
            else:
                print(f"    Timeout at p{plo}/p{phi}, trying tighter box ...")

        print(f"\n  Calibration done.  Selected: p{winning_plo:.0f}/p{winning_phi:.0f}")
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

    # specs 5-7: one per output dim
    for i, (dim, lbl) in enumerate(zip(SPEC_DIMS, SPEC_LABELS)):
        spec_id = 5 + i
        print(f"\n  spec_{spec_id} (Y_{dim}, {lbl}) — PGD ...")
        thr = max(pgd_max(net, lo_t, hi_t, dim, device=device)
                  for net in all_nets.values()) + PGD_MARGIN
        print(f"    threshold Y_{dim} >= {thr:.4f}  (PGD max ~ {thr-PGD_MARGIN:.4f})")
        path = os.path.join(SPEC_DIR, f"spec_{spec_id}.vnnlib")
        write_spec_single(path, lo, hi, dim, thr, winning_plo, winning_phi,
                          f"Spec {spec_id} — Y_{dim} ({lbl}) upper bound")
        spec_paths[spec_id] = path
        thresholds[spec_id] = thr

    # spec_8: OR-clause over all dims
    print(f"\n  spec_8 (any Y_i, OR clause) — PGD all dims ...")
    thr8 = max(pgd_max(net, lo_t, hi_t, d, device=device)
               for net in all_nets.values() for d in range(N_ACT)) + PGD_MARGIN
    print(f"    threshold any Y_i >= {thr8:.4f}  (PGD max ~ {thr8-PGD_MARGIN:.4f})")
    path8 = os.path.join(SPEC_DIR, "spec_8.vnnlib")
    write_spec_or(path8, lo, hi, thr8, winning_plo, winning_phi)
    spec_paths[8] = path8
    thresholds[8] = thr8

    # ── Phase 3: CROWN on baseline ─────────────────────────────────────────────
    crown_results = {}
    if not args.skip_crown:
        print(f"\n{'='*65}")
        print(f"PHASE 3 — CROWN ON BASELINE  (timeout={PHASE3_TIMEOUT}s per spec)")
        print(f"{'='*65}")
        for spec_id, path in sorted(spec_paths.items()):
            print(f"\n  spec_{spec_id} ...", flush=True)
            res, elapsed = run_abcrown(baseline_onnx, path, PHASE3_TIMEOUT)
            crown_results[spec_id] = (res, elapsed)
            print(f"  --> {res}  {elapsed:.1f}s")

    # ── Phase 4: Our method on latent3/latent4 ────────────────────────────────
    ours_results = {}
    if not args.skip_ours:
        print(f"\n{'='*65}")
        print(f"PHASE 4 — OUR METHOD  (latent3/latent4, complete=True)")
        print(f"{'='*65}")
        for label, (rd, quant_step) in BOTTLENECK_RDS.items():
            print(f"\n  {label}  (quant_step={quant_step})")
            for spec_id, path in sorted(spec_paths.items()):
                print(f"    spec_{spec_id} ...", flush=True)
                r = run_ours(rd, quant_step, path, complete=True)
                ours_results[(label, spec_id)] = r
                print(f"    --> {r['result']}  stars={r['n_stars']}  cells={r['n_cells']}  "
                      f"total={r['t_total']:.3f}s")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"COMPARISON SUMMARY — specs 5-8  "
          f"(p{winning_plo:.0f}/p{winning_phi:.0f} nominal box)")
    print(f"{'='*90}")
    print(f"  {'Controller':<10} {'Spec':<8} {'CROWN':>10} {'CROWNt':>8}  "
          f"{'Ours':>8} {'Stars':>6} {'Cells':>8} {'OursT':>8}  {'Speedup':>9}")
    print(f"  {'─'*10} {'─'*8} {'─'*10} {'─'*8}  {'─'*8} {'─'*6} {'─'*8} {'─'*8}  {'─'*9}")

    for spec_id in sorted(spec_paths):
        crown_r, crown_t = crown_results.get(spec_id, ("N/A", 0))
        print(f"  {'baseline':<10} spec_{spec_id:<3} {crown_r:>10} {crown_t:>8.1f}  "
              f"{'N/A':>8} {'N/A':>6} {'N/A':>8} {'N/A':>8}  {'N/A':>9}")
        for label in BOTTLENECK_RDS:
            r = ours_results.get((label, spec_id))
            if r is None:
                continue
            speedup = (f"{crown_t / r['t_total']:.1f}×"
                       if crown_r == "safe" and r["t_total"] > 0 else "N/A")
            print(f"  {label:<10} spec_{spec_id:<3} {'':>10} {'':>8}  "
                  f"{r['result']:>8} {r['n_stars']:>6} {r['n_cells']:>8} "
                  f"{r['t_total']:>8.3f}  {speedup:>9}")

    print(f"\nSpecs written to {SPEC_DIR}/spec_5.vnnlib .. spec_8.vnnlib")


if __name__ == "__main__":
    main()

"""
run_paper_results.py

Reproduce ALL paper results from scratch with full logging.
Runs three methods sequentially (never in parallel) for fair timing:
  1. Our method (encoder nnenum + quantized lookup)
  2. alpha-beta CROWN on full network
  3. nnenum (set_control_settings) on full network

For each (env, controller, spec, method), saves:
  - Result + time to final_results/<method>_results.json
  - Full stdout/stderr log to final_results/logs/<method>_<env>_<ctrl>_spec<N>.log

Usage:
    # Dry run: spec 1 only
    python run_paper_results.py --specs 1

    # Full run: specs 1-10
    python run_paper_results.py --specs 1-10

    # Resume (skips already-completed entries in JSON)
    python run_paper_results.py --specs 1-10
"""
import os, sys, json, argparse, traceback, subprocess, tempfile

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "final_results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

TIMEOUT = 600

# ── Jobs ─────────────────────────────────────────────────────────────────────

JOBS = [
    {
        "env": "HalfCheetah-v4",
        "controllers": [
            {
                "label": "baseline",
                "full_onnx": "sac_sweep_runs/HalfCheetah-v4/baseline/seed0/full_network.onnx",
                "is_baseline": True,
            },
            {
                "label": "latent3",
                "full_onnx": "sac_sweep_runs/HalfCheetah-v4/latent3/seed0/full_network.onnx",
                "encoder_onnx": "sac_sweep_runs/HalfCheetah-v4/latent3/seed0/encoder.onnx",
                "ctrl_pth": "sac_sweep_runs/HalfCheetah-v4/latent3/seed0/latent_controller_full.pth",
                "quant_step": 0.05,
                "is_baseline": False,
            },
        ],
    },
    {
        "env": "Hopper-v5",
        "controllers": [
            {
                "label": "baseline",
                "full_onnx": "sac_sweep_runs/Hopper-v5/baseline512/seed0/full_network.onnx",
                "is_baseline": True,
            },
            {
                "label": "latent3",
                "full_onnx": "sac_sweep_runs/Hopper-v5/latent3/seed0/full_network.onnx",
                "encoder_onnx": "sac_sweep_runs/Hopper-v5/latent3/seed0/encoder.onnx",
                "ctrl_pth": "sac_sweep_runs/Hopper-v5/latent3/seed0/latent_controller_full.pth",
                "quant_step": 0.1,
                "is_baseline": False,
            },
        ],
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def result_key(env, label, spec_id):
    return f"{env}/{label}/spec_{spec_id}"


# ── Method 1: Our method ────────────────────────────────────────────────────

def run_ours(env, ctrl, spec_id, log_path):
    """Run encoder nnenum + quantized lookup in a subprocess to avoid
    fd-redirect / forked-timer deadlocks. Returns (result, time, details)."""
    enc_onnx = os.path.join(BASE_DIR, ctrl["encoder_onnx"])
    full_onnx = os.path.join(BASE_DIR, ctrl["full_onnx"])
    ctrl_pth = os.path.join(BASE_DIR, ctrl["ctrl_pth"])
    spec_path = os.path.join(BASE_DIR, "specs", env, f"spec_{spec_id}.vnnlib")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_json = tmp.name

    worker = os.path.join(BASE_DIR, "_verify_worker.py")
    cmd = [
        sys.executable, worker,
        enc_onnx, spec_path, ctrl_pth,
        str(ctrl["quant_step"]), out_json, full_onnx,
    ]

    with open(log_path, "w") as log_f:
        log_f.write(f"=== Our method: {env} {ctrl['label']} spec_{spec_id} ===\n")
        log_f.write(f"encoder: {ctrl['encoder_onnx']}\n")
        log_f.write(f"controller: {ctrl['ctrl_pth']}\n")
        log_f.write(f"quant_step: {ctrl['quant_step']}\n")
        log_f.write(f"spec: specs/{env}/spec_{spec_id}.vnnlib\n\n")
        log_f.flush()

        proc = subprocess.run(cmd, stdout=log_f, stderr=log_f, cwd=BASE_DIR)

    if proc.returncode != 0:
        raise RuntimeError(f"_verify_worker.py exited with code {proc.returncode}, see {log_path}")

    with open(out_json) as f:
        result = json.load(f)
    os.unlink(out_json)
    return result


# ── Method 2: alpha-beta CROWN ──────────────────────────────────────────────

def run_crown(env, ctrl, spec_id, log_path):
    """Run alpha-beta CROWN in a subprocess to avoid GPU OOM accumulation."""
    onnx_path = os.path.abspath(os.path.join(BASE_DIR, ctrl["full_onnx"]))
    spec_path = os.path.abspath(os.path.join(BASE_DIR, "specs", env, f"spec_{spec_id}.vnnlib"))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_json = tmp.name

    worker = os.path.join(BASE_DIR, "_crown_worker.py")
    cmd = [sys.executable, worker, onnx_path, spec_path, str(TIMEOUT), out_json]

    with open(log_path, "w") as log_f:
        log_f.write(f"=== alpha-beta CROWN: {env} {ctrl['label']} spec_{spec_id} ===\n")
        log_f.write(f"onnx: {ctrl['full_onnx']}\n")
        log_f.write(f"spec: specs/{env}/spec_{spec_id}.vnnlib\n")
        log_f.write(f"timeout: {TIMEOUT}\n\n")
        log_f.flush()

        proc = subprocess.run(cmd, stdout=log_f, stderr=log_f, cwd=BASE_DIR)

    if proc.returncode != 0:
        raise RuntimeError(f"_crown_worker.py exited with code {proc.returncode}, see {log_path}")

    with open(out_json) as f:
        result = json.load(f)
    os.unlink(out_json)
    return result


# ── Method 3: nnenum (control settings) on full network ─────────────────────

def run_nnenum(env, ctrl, spec_id, log_path):
    """Run nnenum in a subprocess to avoid accumulated global state / zombie workers."""
    onnx_path = os.path.join(BASE_DIR, ctrl["full_onnx"])
    spec_path = os.path.join(BASE_DIR, "specs", env, f"spec_{spec_id}.vnnlib")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_json = tmp.name

    worker = os.path.join(BASE_DIR, "_nnenum_worker.py")
    cmd = [sys.executable, worker, onnx_path, spec_path, str(TIMEOUT), out_json]

    with open(log_path, "w") as log_f:
        log_f.write(f"=== nnenum (control): {env} {ctrl['label']} spec_{spec_id} ===\n")
        log_f.write(f"onnx: {ctrl['full_onnx']}\n")
        log_f.write(f"spec: specs/{env}/spec_{spec_id}.vnnlib\n")
        log_f.write(f"settings: set_control_settings(), NUM_PROCESSES=32, TIMEOUT={TIMEOUT}\n\n")
        log_f.flush()

        proc = subprocess.run(cmd, stdout=log_f, stderr=log_f, cwd=BASE_DIR)

    if proc.returncode != 0:
        raise RuntimeError(f"_nnenum_worker.py exited with code {proc.returncode}, see {log_path}")

    with open(out_json) as f:
        result = json.load(f)
    os.unlink(out_json)
    return result


# ── Main loop ────────────────────────────────────────────────────────────────

def parse_specs(s):
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True, help="Spec range, e.g. '1' or '1-10' or '1,5,9'")
    args = ap.parse_args()
    spec_ids = parse_specs(args.specs)

    ours_file = os.path.join(RESULTS_DIR, "ours_results.json")
    crown_file = os.path.join(RESULTS_DIR, "crown_results.json")
    nnenum_file = os.path.join(RESULTS_DIR, "nnenum_results.json")

    ours_data = load_json(ours_file)
    crown_data = load_json(crown_file)
    nnenum_data = load_json(nnenum_file)

    for job in JOBS:
        env = job["env"]
        for ctrl in job["controllers"]:
            label = ctrl["label"]

            for sid in spec_ids:
                spec_path = os.path.join(BASE_DIR, "specs", env, f"spec_{sid}.vnnlib")
                if not os.path.exists(spec_path):
                    print(f"  SKIP {env}/{label}/spec_{sid} — spec file missing")
                    continue

                rkey = result_key(env, label, sid)
                spec_log_dir = os.path.join(LOG_DIR, f"spec_{sid}")
                os.makedirs(spec_log_dir, exist_ok=True)

                # ── Our method (only for bottleneck controllers) ─────────
                if not ctrl["is_baseline"]:
                    if rkey in ours_data:
                        print(f"  [ours]   {rkey}: {ours_data[rkey]['result']} "
                              f"{ours_data[rkey]['time']}s  [cached]")
                    else:
                        log_path = os.path.join(spec_log_dir, f"ours_{env}_{label}.log")
                        print(f"  [ours]   {rkey} ... ", end="", flush=True)
                        try:
                            r = run_ours(env, ctrl, sid, log_path)
                            ours_data[rkey] = r
                            save_json(ours_file, ours_data)
                            print(f"{r['result']}  {r['time']}s  "
                                  f"(stars={r['n_stars']}, cells={r['n_cells']})")
                        except Exception as e:
                            msg = f"error: {e}"
                            ours_data[rkey] = {"result": "error", "time": 0, "error": str(e)}
                            save_json(ours_file, ours_data)
                            print(msg)
                            with open(log_path, "a") as f:
                                f.write(f"\n--- EXCEPTION ---\n{traceback.format_exc()}\n")

                # ── alpha-beta CROWN ─────────────────────────────────────
                if rkey in crown_data:
                    print(f"  [crown]  {rkey}: {crown_data[rkey]['result']} "
                          f"{crown_data[rkey]['time']}s  [cached]")
                else:
                    log_path = os.path.join(spec_log_dir, f"crown_{env}_{label}.log")
                    print(f"  [crown]  {rkey} ... ", end="", flush=True)
                    try:
                        r = run_crown(env, ctrl, sid, log_path)
                        crown_data[rkey] = r
                        save_json(crown_file, crown_data)
                        print(f"{r['result']}  {r['time']}s")
                    except Exception as e:
                        msg = f"error: {e}"
                        crown_data[rkey] = {"result": "error", "time": 0, "error": str(e)}
                        save_json(crown_file, crown_data)
                        print(msg)
                        with open(log_path, "a") as f:
                            f.write(f"\n--- EXCEPTION ---\n{traceback.format_exc()}\n")

                # ── nnenum (control settings) ────────────────────────────
                if rkey in nnenum_data:
                    print(f"  [nnenum] {rkey}: {nnenum_data[rkey]['result']} "
                          f"{nnenum_data[rkey]['time']}s  [cached]")
                else:
                    log_path = os.path.join(spec_log_dir, f"nnenum_{env}_{label}.log")
                    print(f"  [nnenum] {rkey} ... ", end="", flush=True)
                    try:
                        r = run_nnenum(env, ctrl, sid, log_path)
                        nnenum_data[rkey] = r
                        save_json(nnenum_file, nnenum_data)
                        print(f"{r['result']}  {r['time']}s")
                    except Exception as e:
                        msg = f"error: {e}"
                        nnenum_data[rkey] = {"result": "error", "time": 0, "error": str(e)}
                        save_json(nnenum_file, nnenum_data)
                        print(msg)
                        with open(log_path, "a") as f:
                            f.write(f"\n--- EXCEPTION ---\n{traceback.format_exc()}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Results saved to:")
    print(f"    {ours_file}")
    print(f"    {crown_file}")
    print(f"    {nnenum_file}")
    print(f"  Logs in: {LOG_DIR}/")

    all_keys = sorted(set(list(ours_data.keys()) + list(crown_data.keys()) + list(nnenum_data.keys())))
    print(f"\n  {'Key':<45} {'Ours':>12} {'CROWN':>12} {'nnenum':>12}")
    print(f"  {'─'*45} {'─'*12} {'─'*12} {'─'*12}")
    for k in all_keys:
        o = ours_data.get(k, {})
        c = crown_data.get(k, {})
        n = nnenum_data.get(k, {})
        o_str = f"{o.get('result','—')} ({o.get('time','—')}s)" if o else "—"
        c_str = f"{c.get('result','—')} ({c.get('time','—')}s)" if c else "—"
        n_str = f"{n.get('result','—')} ({n.get('time','—')}s)" if n else "—"
        print(f"  {k:<45} {o_str:>12} {c_str:>12} {n_str:>12}")


if __name__ == "__main__":
    main()

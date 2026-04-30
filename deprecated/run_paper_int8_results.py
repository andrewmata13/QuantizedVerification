"""
run_paper_int8_results.py

Reproduce QAT INT8 verification results for the paper.
Runs our method (encoder nnenum + quantized lookup) with the INT8 ORT controller.

Usage:
    python run_paper_int8_results.py --specs 1
    python run_paper_int8_results.py --specs 1-10
"""
import os, sys, json, argparse, traceback, subprocess, tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "int8_results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

JOBS = [
    {
        "env": "HalfCheetah-v4",
        "label": "latent3",
        "full_onnx": "sac_sweep_runs/HalfCheetah-v4/latent3/seed0/full_network.onnx",
        "encoder_onnx": "sac_sweep_runs/HalfCheetah-v4/latent3/seed0/encoder.onnx",
        "ctrl_pth": "sac_sweep_runs/HalfCheetah-v4/latent3/seed0/latent_controller_full.pth",
        "int8_onnx": "sac_sweep_runs/HalfCheetah-v4/latent3/seed0/latent_controller_qat_static_int8.onnx",
        "quant_step": 0.05,
    },
    {
        "env": "Hopper-v5",
        "label": "latent3",
        "full_onnx": "sac_sweep_runs/Hopper-v5/latent3/seed0/full_network.onnx",
        "encoder_onnx": "sac_sweep_runs/Hopper-v5/latent3/seed0/encoder.onnx",
        "ctrl_pth": "sac_sweep_runs/Hopper-v5/latent3/seed0/latent_controller_full.pth",
        "int8_onnx": "sac_sweep_runs/Hopper-v5/latent3/seed0/latent_controller_qat_static_int8.onnx",
        "quant_step": 0.1,
    },
]


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


def run_ours_int8(job, spec_id, log_path):
    enc_onnx = os.path.join(BASE_DIR, job["encoder_onnx"])
    full_onnx = os.path.join(BASE_DIR, job["full_onnx"])
    int8_onnx = os.path.join(BASE_DIR, job["int8_onnx"])
    spec_path = os.path.join(BASE_DIR, "specs", job["env"], f"spec_{spec_id}.vnnlib")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_json = tmp.name

    worker = os.path.join(BASE_DIR, "_verify_int8_worker.py")
    cmd = [
        sys.executable, worker,
        enc_onnx, spec_path, int8_onnx,
        str(job["quant_step"]), out_json, full_onnx,
    ]

    with open(log_path, "w") as log_f:
        log_f.write(f"=== Our method (INT8): {job['env']} {job['label']} spec_{spec_id} ===\n")
        log_f.write(f"encoder: {job['encoder_onnx']}\n")
        log_f.write(f"int8_controller: {job['int8_onnx']}\n")
        log_f.write(f"quant_step: {job['quant_step']}\n")
        log_f.write(f"spec: specs/{job['env']}/spec_{spec_id}.vnnlib\n\n")
        log_f.flush()

        proc = subprocess.run(cmd, stdout=log_f, stderr=log_f, cwd=BASE_DIR)

    if proc.returncode != 0:
        raise RuntimeError(f"_verify_int8_worker.py exited with code {proc.returncode}, see {log_path}")

    with open(out_json) as f:
        result = json.load(f)
    os.unlink(out_json)
    return result


def parse_specs(s):
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    args = ap.parse_args()
    spec_ids = parse_specs(args.specs)

    results_file = os.path.join(RESULTS_DIR, "ours_int8_results.json")
    results_data = load_json(results_file)

    for job in JOBS:
        env = job["env"]
        label = job["label"]
        int8_path = os.path.join(BASE_DIR, job["int8_onnx"])

        if not os.path.exists(int8_path):
            print(f"  SKIP {env}/{label} — no INT8 model at {job['int8_onnx']}")
            continue

        for sid in spec_ids:
            spec_path = os.path.join(BASE_DIR, "specs", env, f"spec_{sid}.vnnlib")
            if not os.path.exists(spec_path):
                print(f"  SKIP {env}/{label}/spec_{sid} — spec file missing")
                continue

            rkey = result_key(env, label, sid)
            spec_log_dir = os.path.join(LOG_DIR, f"spec_{sid}")
            os.makedirs(spec_log_dir, exist_ok=True)

            if rkey in results_data:
                print(f"  [int8]  {rkey}: {results_data[rkey]['result']} "
                      f"{results_data[rkey]['time']}s  [cached]")
                continue

            log_path = os.path.join(spec_log_dir, f"ours_int8_{env}_{label}.log")
            print(f"  [int8]  {rkey} ... ", end="", flush=True)
            try:
                r = run_ours_int8(job, sid, log_path)
                results_data[rkey] = r
                save_json(results_file, results_data)
                print(f"{r['result']}  {r['time']}s  "
                      f"(stars={r['n_stars']}, cells={r['n_cells']})")
            except Exception as e:
                msg = f"error: {e}"
                results_data[rkey] = {"result": "error", "time": 0, "error": str(e)}
                save_json(results_file, results_data)
                print(msg)
                with open(log_path, "a") as f:
                    f.write(f"\n--- EXCEPTION ---\n{traceback.format_exc()}\n")

    # ── 1M-point throughput benchmark + space reduction ────────────────────
    benchmark_file = os.path.join(RESULTS_DIR, "benchmark.json")
    benchmark_data = load_json(benchmark_file)

    for job in JOBS:
        env = job["env"]
        int8_path = os.path.join(BASE_DIR, job["int8_onnx"])
        if not os.path.exists(int8_path):
            continue

        if env in benchmark_data:
            print(f"  [bench] {env}: [cached]")
            continue

        run_dir = os.path.join(BASE_DIR, os.path.dirname(job["encoder_onnx"]))
        print(f"  [bench] {env} — 1M-point throughput + reward eval ... ", end="", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_json = tmp.name

        worker = os.path.join(BASE_DIR, "_benchmark_worker.py")
        cmd = [
            sys.executable, worker,
            os.path.join(BASE_DIR, job["encoder_onnx"]),
            int8_path,
            os.path.join(BASE_DIR, job["ctrl_pth"]),
            str(job["quant_step"]),
            out_json,
            env,
            run_dir,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
        if proc.returncode != 0:
            print(f"error: {proc.stderr[:200]}")
            continue

        with open(out_json) as f:
            bench = json.load(f)
        os.unlink(out_json)
        benchmark_data[env] = bench
        save_json(benchmark_file, benchmark_data)
        tp = bench["throughput"]
        rw = bench["reward"]
        print(f"f32={tp['f32_bottleneck_sec']}s  int8={tp['int8_bottleneck_sec']}s  "
              f"speedup={tp['speedup']}x  reward={rw['mean']}±{rw['std']}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  INT8 RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Results saved to: {results_file}")
    print(f"  Benchmark saved to: {benchmark_file}")
    print(f"  Logs in: {LOG_DIR}/")
    print(f"\n  {'Key':<40} {'Result':>8} {'Time':>8}")
    print(f"  {'─'*40} {'─'*8} {'─'*8}")
    for k in sorted(results_data.keys()):
        r = results_data[k]
        print(f"  {k:<40} {r.get('result','—'):>8} {r.get('time','—'):>8}s")

    if benchmark_data:
        print(f"\n  {'Env':<20} {'f32(1M)':>10} {'int8(1M)':>10} {'Speedup':>8} "
              f"{'f32 size':>10} {'int8 size':>10} {'SizeRatio':>10} {'Reward':>16}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*16}")
        for env, b in sorted(benchmark_data.items()):
            tp = b["throughput"]
            fs = b["file_sizes_bytes"]
            rw = b["reward"]
            print(f"  {env:<20} {tp['f32_bottleneck_sec']:>9.4f}s {tp['int8_bottleneck_sec']:>9.4f}s "
                  f"{tp['speedup']:>7.2f}x {fs['f32_total']:>10,} {fs['int8_total']:>10,} "
                  f"{fs['size_reduction_ratio']:>9.2f}x "
                  f"{rw['mean']}±{rw['std']}")


if __name__ == "__main__":
    main()

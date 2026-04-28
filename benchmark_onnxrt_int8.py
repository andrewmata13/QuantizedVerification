"""
Benchmark ONNX Runtime INT8 dynamic quantization vs float32 for the
HalfCheetah latent3 controller. This uses native int8 GEMM kernels
for actual inference speedup, not just storage savings.
"""
import os, time, torch, numpy as np
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

RUN_DIR = "sac_sweep_runs/HalfCheetah-v4/latent3/seed0"
N_POINTS = 100_000
N_WARMUP = 5
N_TRIALS = 20


def export_controller_onnx(run_dir, out_path):
    ctrl = torch.load(os.path.join(run_dir, "latent_controller_full.pth"),
                      weights_only=False).cpu().eval()
    dummy = torch.randn(1, 3, dtype=torch.float32)
    torch.onnx.export(
        ctrl, dummy, out_path,
        input_names=["latent"],
        output_names=["action"],
        dynamic_axes={"latent": {0: "batch"}, "action": {0: "batch"}},
        opset_version=13,
    )
    return out_path


def benchmark_ort(session, x_np, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    input_name = session.get_inputs()[0].name
    for _ in range(n_warmup):
        session.run(None, {input_name: x_np})
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        session.run(None, {input_name: x_np})
        times.append(time.perf_counter() - t0)
    return times


def benchmark_pytorch(model, x_t, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x_t)
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            model(x_t)
            times.append(time.perf_counter() - t0)
    return times


def main():
    # Export controller to ONNX
    onnx_f32 = os.path.join(RUN_DIR, "latent_controller.onnx")
    onnx_i8 = os.path.join(RUN_DIR, "latent_controller_int8.onnx")

    print("Exporting controller to ONNX...")
    export_controller_onnx(RUN_DIR, onnx_f32)
    print(f"  float32 ONNX: {os.path.getsize(onnx_f32):,} bytes "
          f"({os.path.getsize(onnx_f32)/1024:.1f} KB)")

    # Quantize with ONNX Runtime (dynamic INT8)
    print("Quantizing to INT8 (ONNX Runtime dynamic)...")
    quantize_dynamic(
        onnx_f32, onnx_i8,
        weight_type=QuantType.QInt8,
    )
    print(f"  INT8 ONNX:    {os.path.getsize(onnx_i8):,} bytes "
          f"({os.path.getsize(onnx_i8)/1024:.1f} KB)")
    print(f"  Compression:  {os.path.getsize(onnx_f32)/os.path.getsize(onnx_i8):.2f}x")

    # Create sessions
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1

    sess_f32 = ort.InferenceSession(onnx_f32, opts, providers=["CPUExecutionProvider"])
    sess_i8 = ort.InferenceSession(onnx_i8, opts, providers=["CPUExecutionProvider"])

    # Also create multi-threaded sessions
    opts_mt = ort.SessionOptions()
    opts_mt.intra_op_num_threads = 0  # use all cores
    opts_mt.inter_op_num_threads = 0

    sess_f32_mt = ort.InferenceSession(onnx_f32, opts_mt, providers=["CPUExecutionProvider"])
    sess_i8_mt = ort.InferenceSession(onnx_i8, opts_mt, providers=["CPUExecutionProvider"])

    # Load PyTorch model for comparison
    ctrl_pt = torch.load(os.path.join(RUN_DIR, "latent_controller_full.pth"),
                         weights_only=False).cpu().eval()

    # Generate test data
    x_np = np.random.randn(N_POINTS, 3).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    # ── Output correctness ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Output Correctness")
    print(f"{'='*60}")

    out_f32 = sess_f32.run(None, {"latent": x_np})[0]
    out_i8 = sess_i8.run(None, {"latent": x_np})[0]
    with torch.no_grad():
        out_pt = ctrl_pt(x_t).numpy()

    diff_ort = np.abs(out_f32 - out_i8)
    diff_pt = np.abs(out_pt - out_f32)

    print(f"  ONNX f32 vs PyTorch f32:  max={diff_pt.max():.8f}  mean={diff_pt.mean():.8f}")
    print(f"  ONNX f32 vs ONNX INT8:    max={diff_ort.max():.6f}  mean={diff_ort.mean():.6f}")

    # ── Benchmark ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Inference Time ({N_POINTS:,} points, {N_TRIALS} trials)")
    print(f"{'='*60}")

    # Single-threaded
    t_pt = benchmark_pytorch(ctrl_pt, x_t)
    t_f32_1t = benchmark_ort(sess_f32, x_np)
    t_i8_1t = benchmark_ort(sess_i8, x_np)

    med_pt = np.median(t_pt)
    med_f32_1t = np.median(t_f32_1t)
    med_i8_1t = np.median(t_i8_1t)

    print(f"\n  Single-threaded CPU:")
    print(f"    PyTorch f32:    {med_pt*1000:.1f} ms")
    print(f"    ORT f32:        {med_f32_1t*1000:.1f} ms  "
          f"({med_pt/med_f32_1t:.2f}x vs PyTorch)")
    print(f"    ORT INT8:       {med_i8_1t*1000:.1f} ms  "
          f"({med_f32_1t/med_i8_1t:.2f}x vs ORT f32, "
          f"{med_pt/med_i8_1t:.2f}x vs PyTorch)")

    # Multi-threaded
    t_f32_mt = benchmark_ort(sess_f32_mt, x_np)
    t_i8_mt = benchmark_ort(sess_i8_mt, x_np)

    med_f32_mt = np.median(t_f32_mt)
    med_i8_mt = np.median(t_i8_mt)

    print(f"\n  Multi-threaded CPU:")
    print(f"    ORT f32:        {med_f32_mt*1000:.1f} ms  "
          f"({med_pt/med_f32_mt:.2f}x vs PyTorch)")
    print(f"    ORT INT8:       {med_i8_mt*1000:.1f} ms  "
          f"({med_f32_mt/med_i8_mt:.2f}x vs ORT f32, "
          f"{med_pt/med_i8_mt:.2f}x vs PyTorch)")

    # ── Batch size sweep ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Batch Size Sweep (single-threaded)")
    print(f"{'='*60}")
    print(f"  {'Batch':>10} {'ORT f32 (ms)':>14} {'ORT INT8 (ms)':>14} {'Speedup':>10}")
    print(f"  {'─'*10} {'─'*14} {'─'*14} {'─'*10}")

    for batch in [1, 10, 100, 1000, 10000, 100000]:
        x_b = np.random.randn(batch, 3).astype(np.float32)
        tf = benchmark_ort(sess_f32, x_b, n_warmup=3, n_trials=10)
        ti = benchmark_ort(sess_i8, x_b, n_warmup=3, n_trials=10)
        mf = np.median(tf) * 1000
        mi = np.median(ti) * 1000
        print(f"  {batch:>10,} {mf:>14.3f} {mi:>14.3f} {mf/mi:>10.2f}x")


if __name__ == "__main__":
    main()

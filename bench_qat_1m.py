"""Benchmark QAT static INT8 vs float32, 1M points."""
import os, time, torch, numpy as np
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import onnxruntime as ort

RD = "sac_sweep_runs/HalfCheetah-v4/latent3/seed0"
N = 1_000_000
NW, NT = 5, 15


def bench_ort(sess, x):
    inp = sess.get_inputs()[0].name
    for _ in range(NW):
        sess.run(None, {inp: x})
    times = []
    for _ in range(NT):
        t0 = time.perf_counter()
        sess.run(None, {inp: x})
        times.append(time.perf_counter() - t0)
    return times


def bench_pt(model, x):
    with torch.no_grad():
        for _ in range(NW):
            model(x)
        times = []
        for _ in range(NT):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    return times


x_np = np.random.randn(N, 3).astype(np.float32)
x_t = torch.from_numpy(x_np)

# ── PyTorch f32 ──────────────────────────────────────────────────────────
ctrl = torch.load(f"{RD}/latent_controller_full.pth", weights_only=False).cpu().eval()
t_pt = bench_pt(ctrl, x_t)
del ctrl
med_pt = np.median(t_pt)
print(f"PyTorch f32:          {med_pt*1000:8.1f} ms  (median of {NT})")

# ── ORT f32 single-threaded ─────────────────────────────────────────────
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
sess = ort.InferenceSession(f"{RD}/latent_controller.onnx", opts,
                            providers=["CPUExecutionProvider"])
t_f32_1t = bench_ort(sess, x_np)
med_f32_1t = np.median(t_f32_1t)
print(f"ORT f32 (1 thread):   {med_f32_1t*1000:8.1f} ms")
del sess

# ── ORT QAT INT8 single-threaded ────────────────────────────────────────
sess = ort.InferenceSession(f"{RD}/latent_controller_qat_static_int8.onnx", opts,
                            providers=["CPUExecutionProvider"])
t_i8_1t = bench_ort(sess, x_np)
med_i8_1t = np.median(t_i8_1t)
print(f"ORT QAT INT8 (1 thr): {med_i8_1t*1000:8.1f} ms  "
      f"({med_f32_1t/med_i8_1t:.2f}x vs ORT f32)")
del sess

# ── ORT f32 multi-threaded ──────────────────────────────────────────────
opts_mt = ort.SessionOptions()
opts_mt.intra_op_num_threads = 0
opts_mt.inter_op_num_threads = 0
sess = ort.InferenceSession(f"{RD}/latent_controller.onnx", opts_mt,
                            providers=["CPUExecutionProvider"])
t_f32_mt = bench_ort(sess, x_np)
med_f32_mt = np.median(t_f32_mt)
print(f"ORT f32 (all cores):  {med_f32_mt*1000:8.1f} ms  "
      f"({med_pt/med_f32_mt:.2f}x vs PyTorch)")
del sess

# ── ORT QAT INT8 multi-threaded ─────────────────────────────────────────
sess = ort.InferenceSession(f"{RD}/latent_controller_qat_static_int8.onnx", opts_mt,
                            providers=["CPUExecutionProvider"])
t_i8_mt = bench_ort(sess, x_np)
med_i8_mt = np.median(t_i8_mt)
print(f"ORT QAT INT8 (all):   {med_i8_mt*1000:8.1f} ms  "
      f"({med_f32_mt/med_i8_mt:.2f}x vs ORT f32, "
      f"{med_pt/med_i8_mt:.2f}x vs PyTorch)")
del sess

print(f"\nSummary ({N:,} points, median of {NT} trials):")
print(f"  Storage: 3.72x compression")
print(f"  Policy:  13,530 vs 13,538 return (QAT INT8 vs f32)")

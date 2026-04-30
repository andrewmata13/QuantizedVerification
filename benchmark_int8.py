"""
INT8 weight quantization benchmark for HalfCheetah latent3 controller.

1. Memory footprint: float32 vs INT8
2. Execution time: encoder, rounding, controller (f32 via PyTorch, INT8 via ORT) on 1M points
3. Output difference: float32 vs INT8
"""
import os, time, tempfile, json, numpy as np
import onnx
import onnxruntime as ort

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MUJOCO_GL"] = "egl"

RUN_DIR    = "sac_sweep_runs/HalfCheetah-v4/latent3/seed0"
QUANT_STEP = 0.05
N_POINTS   = 1_000_000
N_WARMUP   = 5
N_TRIALS   = 10


def add_batch_dim(path):
    model = onnx.load(path)
    inp_shape = model.graph.input[0].type.tensor_type.shape
    if len(inp_shape.dim) >= 2:
        return path
    for tensor in list(model.graph.input) + list(model.graph.output):
        shape = tensor.type.tensor_type.shape
        new_dim = onnx.TensorShapeProto.Dimension()
        new_dim.dim_param = "batch"
        shape.dim.insert(0, new_dim)
    del model.graph.value_info[:]
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, tmp.name)
    return tmp.name


def make_session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def benchmark_ort(sess, x_np, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    inp_name = sess.get_inputs()[0].name
    for _ in range(n_warmup):
        sess.run(None, {inp_name: x_np})
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: x_np})
        times.append(time.perf_counter() - t0)
    return times


def benchmark_rounding(z_np, quant_step, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    for _ in range(n_warmup):
        np.round(z_np / quant_step) * quant_step
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        np.round(z_np / quant_step) * quant_step
        times.append(time.perf_counter() - t0)
    return times


def main():
    print(f"Benchmarking {RUN_DIR}")

    enc_onnx = os.path.join(RUN_DIR, "encoder.onnx")
    f32_onnx = os.path.join(RUN_DIR, "latent_controller.onnx")
    int8_onnx = os.path.join(RUN_DIR, "latent_controller_int8.onnx")

    enc_batched = add_batch_dim(enc_onnx)
    f32_batched = add_batch_dim(f32_onnx)
    int8_batched = add_batch_dim(int8_onnx)

    enc_sess = make_session(enc_batched)
    f32_sess = make_session(f32_batched)
    int8_sess = make_session(int8_batched)

    n_obs = enc_sess.get_inputs()[0].shape[-1]
    n_latent = enc_sess.get_outputs()[0].shape[-1]
    n_act = int8_sess.get_outputs()[0].shape[-1]

    # ── 1. Memory comparison ─────────────────────────────────────────────────
    enc_size = os.path.getsize(enc_onnx)
    f32_size = os.path.getsize(f32_onnx)
    int8_size = os.path.getsize(int8_onnx)

    print(f"\n{'='*60}")
    print(f"  Memory Footprint")
    print(f"{'='*60}")
    print(f"  Encoder ONNX:       {enc_size:,} B ({enc_size/1024:.1f} KB)")
    print(f"  f32 controller:     {f32_size:,} B ({f32_size/1024:.1f} KB)")
    print(f"  INT8 controller:    {int8_size:,} B ({int8_size/1024:.1f} KB)")
    print(f"  f32 total:          {enc_size + f32_size:,} B")
    print(f"  INT8 total:         {enc_size + int8_size:,} B")
    print(f"  Size reduction:     {(enc_size + f32_size) / (enc_size + int8_size):.2f}x")

    # ── 2. Execution time benchmark ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Execution Time ({N_POINTS:,} points, {N_TRIALS} trials)")
    print(f"{'='*60}")

    obs_np = np.random.default_rng(42).standard_normal((N_POINTS, n_obs)).astype(np.float32)

    # Encoder (ORT)
    t_enc = benchmark_ort(enc_sess, obs_np)
    med_enc = np.median(t_enc)
    print(f"\n  Encoder (ORT):")
    print(f"    {med_enc*1000:.1f} ms (median)")

    # Rounding
    z_np = enc_sess.run(None, {enc_sess.get_inputs()[0].name: obs_np})[0]
    t_round = benchmark_rounding(z_np, QUANT_STEP)
    med_round = np.median(t_round)
    print(f"\n  Latent rounding (quant_step={QUANT_STEP}):")
    print(f"    {med_round*1000:.1f} ms (median)")

    # f32 controller (ORT)
    t_f32 = benchmark_ort(f32_sess, z_np)
    med_f32 = np.median(t_f32)
    print(f"\n  f32 controller (ORT):")
    print(f"    {med_f32*1000:.1f} ms (median)")

    # INT8 controller (ORT)
    z_q = np.round(z_np / QUANT_STEP) * QUANT_STEP
    t_int8 = benchmark_ort(int8_sess, z_q.astype(np.float32))
    med_int8 = np.median(t_int8)
    print(f"\n  INT8 controller (ORT):")
    print(f"    {med_int8*1000:.1f} ms (median)")
    print(f"    Speedup vs f32: {med_f32/med_int8:.2f}x")

    # Total pipeline
    total_f32 = med_enc + med_round + med_f32
    total_int8 = med_enc + med_round + med_int8
    print(f"\n  Total pipeline (encoder + round + controller):")
    print(f"    f32:  {total_f32*1000:.1f} ms")
    print(f"    INT8: {total_int8*1000:.1f} ms")
    print(f"    Speedup: {total_f32/total_int8:.2f}x")

    # ── Save JSON ────────────────────────────────────────────────────────────
    result = {
        "n_points": N_POINTS,
        "memory": {
            "encoder_bytes": enc_size,
            "f32_controller_bytes": f32_size,
            "int8_controller_bytes": int8_size,
            "f32_total_bytes": enc_size + f32_size,
            "int8_total_bytes": enc_size + int8_size,
            "size_reduction": round((enc_size + f32_size) / (enc_size + int8_size), 2),
        },
        "throughput_ms": {
            "encoder": round(med_enc * 1000, 1),
            "rounding": round(med_round * 1000, 1),
            "controller_f32": round(med_f32 * 1000, 1),
            "controller_int8": round(med_int8 * 1000, 1),
            "controller_speedup": round(med_f32 / med_int8, 2),
            "total_f32": round(total_f32 * 1000, 1),
            "total_int8": round(total_int8 * 1000, 1),
            "total_speedup": round(total_f32 / total_int8, 2),
        },
    }

    out_path = os.path.join("paper_specs", "benchmark_int8.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {out_path}")

    # Cleanup temp files
    for p in [enc_batched, f32_batched, int8_batched]:
        if p not in [enc_onnx, f32_onnx, int8_onnx]:
            os.unlink(p)


if __name__ == "__main__":
    main()

"""
INT8 weight quantization benchmark for HalfCheetah latent3 controller.

1. Execution time: float32 vs INT8 on 1M forward passes
2. Memory footprint comparison
3. Verification on all 10 HC specs with INT8 model
"""
import os, copy, time, torch, torch.nn as nn, numpy as np
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from nnenum.settings import Settings
Settings.CHECK_SINGLE_THREAD_BLAS = False
from nnenum.vnnlib import get_num_inputs_outputs
from verify_policy import verify

RUN_DIR    = "sac_sweep_runs/HalfCheetah-v4/latent3/seed0"
QUANT_STEP = 0.05
N_POINTS   = 1_000_000
N_WARMUP   = 5
N_TRIALS   = 10


def quantize_weights_int8(ctrl):
    ctrl_q = copy.deepcopy(ctrl)
    scales = {}
    for name, module in ctrl_q.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            scale = w.abs().max(dim=1, keepdim=True).values / 127.0
            scale = scale.clamp(min=1e-8)
            w_int8 = (w / scale).round().clamp(-128, 127)
            module.weight.data = (w_int8 * scale).float()
            scales[name] = scale.squeeze()
    return ctrl_q, scales


def benchmark_forward(model, x, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_trials):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return times


def model_size_info(ctrl, scales=None):
    n_weights = sum(m.weight.numel() for m in ctrl.modules() if isinstance(m, nn.Linear))
    n_biases = sum(m.bias.numel() for m in ctrl.modules() if isinstance(m, nn.Linear) and m.bias is not None)
    f32_bytes = sum(p.numel() * p.element_size() for p in ctrl.parameters())
    if scales is not None:
        n_scales = sum(s.numel() for s in scales.values())
        i8_bytes = n_weights * 1 + n_biases * 4 + n_scales * 4
        return n_weights, n_biases, f32_bytes, i8_bytes, n_scales
    return n_weights, n_biases, f32_bytes


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ctrl_path = os.path.join(RUN_DIR, "latent_controller_full.pth")
    enc_onnx = os.path.join(RUN_DIR, "encoder.onnx")
    full_onnx = os.path.join(RUN_DIR, "full_network.onnx")

    ctrl_f32 = torch.load(ctrl_path, weights_only=False).cpu().eval()
    ctrl_i8, scales = quantize_weights_int8(ctrl_f32)

    n_in, _, _ = get_num_inputs_outputs(enc_onnx)
    _, n_act, _ = get_num_inputs_outputs(full_onnx)
    _, n_latent, _ = get_num_inputs_outputs(enc_onnx)

    # ── 1. Memory comparison ─────────────────────────────────────────────────
    n_weights, n_biases, f32_bytes, i8_bytes, n_scales = model_size_info(ctrl_f32, scales)

    print(f"\n{'='*60}")
    print(f"  Memory Footprint")
    print(f"{'='*60}")
    print(f"  Architecture: {n_latent} → 512 → 512 → {n_act}")
    print(f"  Weight parameters: {n_weights:,}")
    print(f"  Bias parameters:   {n_biases:,}")
    print(f"  Quantization scales: {n_scales:,}")
    print(f"")
    print(f"  float32 storage: {f32_bytes:,} B ({f32_bytes/1024:.1f} KB)")
    print(f"    = {n_weights:,} weights x 4B + {n_biases:,} biases x 4B")
    print(f"  INT8 storage:    {i8_bytes:,} B ({i8_bytes/1024:.1f} KB)")
    print(f"    = {n_weights:,} weights x 1B + {n_biases:,} biases x 4B + {n_scales:,} scales x 4B")
    print(f"  Compression:     {f32_bytes/i8_bytes:.2f}x")

    # ── 2. Execution time benchmark ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Execution Time ({N_POINTS:,} points, {N_TRIALS} trials)")
    print(f"{'='*60}")

    x = torch.randn(N_POINTS, n_latent, dtype=torch.float32)

    # CPU benchmark
    ctrl_f32_cpu = ctrl_f32.cpu().eval()
    ctrl_i8_cpu = ctrl_i8.cpu().eval()

    t_f32_cpu = benchmark_forward(ctrl_f32_cpu, x)
    t_i8_cpu = benchmark_forward(ctrl_i8_cpu, x)

    med_f32_cpu = np.median(t_f32_cpu)
    med_i8_cpu = np.median(t_i8_cpu)

    print(f"\n  CPU:")
    print(f"    float32: {med_f32_cpu*1000:.1f} ms (median), "
          f"runs: {[f'{t*1000:.1f}' for t in t_f32_cpu]}")
    print(f"    INT8:    {med_i8_cpu*1000:.1f} ms (median), "
          f"runs: {[f'{t*1000:.1f}' for t in t_i8_cpu]}")
    print(f"    Speedup: {med_f32_cpu/med_i8_cpu:.2f}x")

    if device == "cuda":
        ctrl_f32_gpu = ctrl_f32.to(device).eval()
        ctrl_i8_gpu = ctrl_i8.to(device).eval()

        t_f32_gpu = benchmark_forward(ctrl_f32_gpu, x)
        t_i8_gpu = benchmark_forward(ctrl_i8_gpu, x)

        med_f32_gpu = np.median(t_f32_gpu)
        med_i8_gpu = np.median(t_i8_gpu)

        print(f"\n  GPU ({torch.cuda.get_device_name()}):")
        print(f"    float32: {med_f32_gpu*1000:.2f} ms (median), "
              f"runs: {[f'{t*1000:.2f}' for t in t_f32_gpu]}")
        print(f"    INT8:    {med_i8_gpu*1000:.2f} ms (median), "
              f"runs: {[f'{t*1000:.2f}' for t in t_i8_gpu]}")
        print(f"    Speedup: {med_f32_gpu/med_i8_gpu:.2f}x")

    # ── 3. Output difference ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Output Difference (float32 vs INT8)")
    print(f"{'='*60}")

    ctrl_f32_cpu = ctrl_f32.cpu().eval()
    ctrl_i8_cpu = ctrl_i8.cpu().eval()
    x_cpu = x.cpu()
    with torch.no_grad():
        out_f32 = ctrl_f32_cpu(x_cpu).numpy()
        out_i8 = ctrl_i8_cpu(x_cpu).numpy()

    diff = np.abs(out_f32 - out_i8)
    print(f"  Max abs diff:  {diff.max():.6f}")
    print(f"  Mean abs diff: {diff.mean():.6f}")
    print(f"  Std abs diff:  {diff.std():.6f}")
    print(f"  Max relative:  {(diff / (np.abs(out_f32) + 1e-8)).max():.6f}")

    # ── 4. Verification on INT8 model (specs 1-10) ───────────────────────────
    print(f"\n{'='*60}")
    print(f"  Verification: INT8 vs float32 (specs 1-10)")
    print(f"{'='*60}")
    print(f"  {'Spec':<10} {'f32 result':<12} {'f32 time':>10} {'i8 result':<12} {'i8 time':>10} {'max diff':>10}")
    print(f"  {'─'*10} {'─'*12} {'─'*10} {'─'*12} {'─'*10} {'─'*10}")

    # Save INT8 model for verify to load
    i8_path = os.path.join(RUN_DIR, "latent_controller_int8.pth")
    torch.save(ctrl_i8, i8_path)

    for sid in range(1, 11):
        spec_path = f"specs/HalfCheetah-v4/spec_{sid}.vnnlib"

        r_f32 = verify(enc_onnx, spec_path, ctrl_path,
                       n_inputs=n_in, n_actions=n_act,
                       quant_step=QUANT_STEP, complete=True)
        r_i8 = verify(enc_onnx, spec_path, ctrl_i8,
                      n_inputs=n_in, n_actions=n_act,
                      quant_step=QUANT_STEP, complete=True)

        cf32 = {c: np.array(a) for c, a in r_f32['cells']}
        ci8 = {c: np.array(a) for c, a in r_i8['cells']}
        diffs = [np.abs(cf32[c] - ci8[c]).max() for c in cf32 if c in ci8]
        max_diff = max(diffs) if diffs else 0.0

        print(f"  spec_{sid:<5} {r_f32['result']:<12} {r_f32['t_total']:>9.2f}s "
              f"{r_i8['result']:<12} {r_i8['t_total']:>9.2f}s {max_diff:>10.5f}")

    # ── 5. Quantization alternatives ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Quantization Alternatives Analysis")
    print(f"{'='*60}")

    # INT4 simulation
    ctrl_i4 = copy.deepcopy(ctrl_f32).cpu()
    for name, module in ctrl_i4.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            scale = w.abs().max(dim=1, keepdim=True).values / 7.0
            scale = scale.clamp(min=1e-8)
            w_i4 = (w / scale).round().clamp(-8, 7)
            module.weight.data = (w_i4 * scale).float()
    with torch.no_grad():
        out_i4 = ctrl_i4(x_cpu).numpy()
    diff_i4 = np.abs(out_f32 - out_i4)

    # FP16 simulation
    ctrl_fp16 = copy.deepcopy(ctrl_f32).cpu()
    for name, module in ctrl_fp16.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = module.weight.data.half().float()
            if module.bias is not None:
                module.bias.data = module.bias.data.half().float()
    with torch.no_grad():
        out_fp16 = ctrl_fp16(x_cpu).numpy()
    diff_fp16 = np.abs(out_f32 - out_fp16)

    n_scales_val = sum(s.numel() for s in scales.values())
    i4_bytes = n_weights * 0.5 + n_biases * 4 + n_scales_val * 4
    fp16_bytes = (n_weights + n_biases) * 2

    print(f"\n  {'Format':<12} {'Storage':>10} {'Compression':>12} {'Max diff':>10} {'Mean diff':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*12} {'─'*10} {'─'*10}")
    print(f"  {'float32':<12} {f32_bytes/1024:>9.1f}K {'1.00x':>12} {'0.000000':>10} {'0.000000':>10}")
    print(f"  {'FP16':<12} {fp16_bytes/1024:>9.1f}K {f32_bytes/fp16_bytes:>11.2f}x {diff_fp16.max():>10.6f} {diff_fp16.mean():>10.6f}")
    print(f"  {'INT8':<12} {i8_bytes/1024:>9.1f}K {f32_bytes/i8_bytes:>11.2f}x {diff.max():>10.6f} {diff.mean():>10.6f}")
    print(f"  {'INT4':<12} {i4_bytes/1024:>9.1f}K {f32_bytes/i4_bytes:>11.2f}x {diff_i4.max():>10.6f} {diff_i4.mean():>10.6f}")

    t_fp16_cpu = benchmark_forward(ctrl_fp16.cpu().eval(), x)
    t_i4_cpu = benchmark_forward(ctrl_i4.cpu().eval(), x)
    med_fp16 = np.median(t_fp16_cpu)
    med_i4 = np.median(t_i4_cpu)

    print(f"\n  CPU inference time ({N_POINTS:,} points, median of {N_TRIALS}):")
    print(f"    float32: {med_f32_cpu*1000:.1f} ms")
    print(f"    FP16:    {med_fp16*1000:.1f} ms  ({med_f32_cpu/med_fp16:.2f}x)")
    print(f"    INT8:    {med_i8_cpu*1000:.1f} ms  ({med_f32_cpu/med_i8_cpu:.2f}x)")
    print(f"    INT4:    {med_i4*1000:.1f} ms  ({med_f32_cpu/med_i4:.2f}x)")


if __name__ == "__main__":
    main()

"""
quantize_and_verify.py

Quantize the latent controller to INT8 and run our verification pipeline on it.
Demonstrates that our method verifies the exact deployed (quantized) network
with no additional cost, unlike BaB/CROWN which require MILP reformulation.

Usage:
    python quantize_and_verify.py
"""
import os, copy, torch, torch.nn as nn, numpy as np
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from nnenum.vnnlib import get_num_inputs_outputs
from verify_policy import verify

CONTROLLERS = {
    "latent3": {
        "rd":        "sac_sweep_runs/HalfCheetah-v4/latent3/seed0",
        "quant_step": 0.05,
    },
}
SPECS = [f"specs/HalfCheetah-v4/spec_{i}.vnnlib" for i in range(1, 5)]


def quantize_weights_int8(ctrl_path):
    """
    Per-output-channel symmetric INT8 weight-only quantization.
    Weights are rounded to the nearest INT8 value and dequantized back to
    float32 for inference. Activations remain float32 throughout.
    Returns the dequantized model and a dict of {layer_name: scale_tensor}.
    """
    ctrl = torch.load(ctrl_path, weights_only=False).cpu().eval()
    ctrl_q = copy.deepcopy(ctrl)
    scales = {}
    for name, module in ctrl_q.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data                              # [out, in]
            scale = w.abs().max(dim=1, keepdim=True).values / 127.0
            scale = scale.clamp(min=1e-8)
            w_int8 = (w / scale).round().clamp(-128, 127)
            module.weight.data = (w_int8 * scale).float()
            scales[name] = scale.squeeze()
    return ctrl_q, scales


def model_size_bytes(ctrl):
    """Total bytes used by all parameters (as stored in the model)."""
    return sum(p.numel() * p.element_size() for p in ctrl.parameters())


def weight_param_count(ctrl):
    """Number of weight (not bias) parameters across all Linear layers."""
    return sum(m.weight.numel() for m in ctrl.modules() if isinstance(m, nn.Linear))


for label, cfg in CONTROLLERS.items():
    rd         = cfg["rd"]
    quant_step = cfg["quant_step"]
    ctrl_path  = os.path.join(rd, "latent_controller_full.pth")
    enc_onnx   = os.path.join(rd, "encoder.onnx")
    full_onnx  = os.path.join(rd, "full_network.onnx")
    n_in, _, _ = get_num_inputs_outputs(enc_onnx)
    _, n_act, _ = get_num_inputs_outputs(full_onnx)

    print(f"\n{'='*60}")
    print(f"  {label}  —  INT8 weight quantization")
    print(f"{'='*60}")

    # ── Quantize ──────────────────────────────────────────────────────────────
    ctrl_f32 = torch.load(ctrl_path, weights_only=False).cpu().eval()
    ctrl_i8, scales = quantize_weights_int8(ctrl_path)

    # ── Model size comparison ─────────────────────────────────────────────────
    n_weights   = weight_param_count(ctrl_f32)
    f32_bytes   = model_size_bytes(ctrl_f32)
    # INT8 storage: weights as int8 (1B each) + per-channel scales as float32
    n_scales    = sum(s.numel() for s in scales.values())
    i8_bytes    = n_weights * 1 + n_scales * 4   # int8 weights + float32 scales
    ratio       = f32_bytes / i8_bytes

    print(f"\n  Weight parameters : {n_weights:,}")
    print(f"  float32 storage   : {f32_bytes / 1024:.1f} KB  "
          f"({n_weights:,} × 4 B)")
    print(f"  INT8 storage      : {i8_bytes / 1024:.1f} KB  "
          f"({n_weights:,} × 1 B + {n_scales} scales × 4 B)")
    print(f"  Compression ratio : {ratio:.2f}×")

    # Verify a few weights are actually discretised
    for name, module in ctrl_i8.named_modules():
        if isinstance(module, nn.Linear):
            w_q = module.weight.data
            sc  = scales[name].unsqueeze(1)
            steps = (w_q / sc).round()
            unique_steps = steps[0].unique().numel()
            print(f"  [{name}] row-0 unique INT8 steps: {unique_steps} "
                  f"(max possible 256)")
            break

    # ── Run verification on both float32 and int8 ──────────────────────────────
    print(f"\n  {'Spec':<10} {'Model':<10} {'Result':<8} {'Stars':>6} {'Cells':>8} {'Time(s)':>8}")
    print(f"  {'─'*10} {'─'*10} {'─'*8} {'─'*6} {'─'*8} {'─'*8}")

    all_diffs = []
    for spec_path in SPECS:
        spec_name = os.path.basename(spec_path).replace(".vnnlib", "")

        r_f32 = verify(enc_onnx, spec_path, ctrl_path,
                       n_inputs=n_in, n_actions=n_act,
                       quant_step=quant_step, complete=True)
        r_i8  = verify(enc_onnx, spec_path, ctrl_i8,
                       n_inputs=n_in, n_actions=n_act,
                       quant_step=quant_step, complete=True)

        print(f"  {spec_name:<10} {'float32':<10} {r_f32['result']:<8} "
              f"{r_f32['n_stars']:>6} {r_f32['n_cells']:>8} {r_f32['t_total']:>8.3f}")
        print(f"  {'':<10} {'int8':<10} {r_i8['result']:<8} "
              f"{r_i8['n_stars']:>6} {r_i8['n_cells']:>8} {r_i8['t_total']:>8.3f}")

        cf32  = {c: np.array(a) for c, a in r_f32['cells']}
        ci8   = {c: np.array(a) for c, a in r_i8['cells']}
        diffs = [np.abs(cf32[c] - ci8[c]).max() for c in cf32 if c in ci8]
        all_diffs.extend(diffs)

    all_diffs = np.array(all_diffs)
    print(f"\n  Action diff at reachable cells (float32 vs int8, all specs):")
    print(f"    Max  = {all_diffs.max():.5f}")
    print(f"    Mean = {all_diffs.mean():.5f}")
    print(f"    Std  = {all_diffs.std():.5f}")

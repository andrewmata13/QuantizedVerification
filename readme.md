## Setup

```bash
conda create -n quant-env
conda activate quant-env

conda install -c conda-forge \
    mesa-libgl-cos7-x86_64 \
    mesa-libegl-cos7-x86_64 \
    libglvnd-cos7-x86_64

conda install -c conda-forge cvxpy cvxopt swiglpk
conda install -c gurobi gurobi
conda install -c mosek mosek

pip install -r requirements.txt
pip install --upgrade onnxscript

cd nnenum_package/nnenum
pip install .
cd ../..

cd nnenum_package/iq_verify
pip install .
cd ../..
```

**Note:** nnenum requires single-threaded BLAS. Always run with:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python <script>.py
```

If you see a `CXXABI_1.3.15` error:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## Overview

This project demonstrates a **quantized bottleneck verification** approach for SAC policies trained with a low-dimensional latent bottleneck. The bottleneck splits the policy into two verifiable components:

1. **Encoder** (obs → N-D latent): small network, fast to verify with nnenum
2. **Latent controller** (N-D latent → action): verified by enumerating reachable quantized latent cells

### HalfCheetah-v4 Architecture (latent dim = 1)

```
Observation (17)
    → Linear(17, 16) → ReLU
    → Linear(16, 1)  → ReLU      ← bottleneck (1-D latent)
    → Linear(1, 512) → ReLU
    → Linear(512, 512) → ReLU
    → Linear(512, 6)              ← pre-tanh actions
    → Tanh                        ← final actions (applied at runtime only)
```

nnenum splits each `nn.Linear` into MatMul + Add. The encoder is **6 nnenum layers** (17 ReLU neurons); the full network is **14 nnenum layers** (17 + 1024 ReLU neurons). This asymmetry is the core of the speedup.

---

## Training

```bash
# HalfCheetah-v4, latent dim 1 (arch [16, 1, 512, 512])
python train_custom_sb3.py

# HalfCheetah-v4, latent dims 2 and 3 (for verification scaling comparison)
python train_latent_sweep.py
```

Each run saves to `sac_sweep_runs/<env>/<run_name>/seed0/`:
- `model.zip` — full SB3 SAC model
- `encoder.onnx` — encoder network (ONNX, ReLU-only, no Tanh)
- `encoder_full.pth`, `latent_controller_full.pth` — PyTorch modules
- `train_vec_norm.pkl` — VecNormalize running statistics

### Policy Performance (HalfCheetah-v4, 10 episodes)

"Quantized" uses quant_step=0.005 (same as verification). Returns are near-identical to the clean controller, confirming the quantization grid is fine enough to not degrade policy quality.

| Architecture | Clean return | Quantized return (q=0.005) |
|---|---|---|
| `[16, 1, 512, 512]` | 6683 ± 98 | 6683 ± 60 |
| `[16, 2, 512, 512]` | 8705 ± 1644 | 9276 ± 79 |
| `[16, 3, 512, 512]` | 13522 ± 59 | 13506 ± 62 |

---

## Verification

### Approach

Specs are written in VNN-LIB format with:
- **Inputs** `X_0..X_16`: VecNormalize-normalized observations (computed from rollout percentiles for a given behavioral regime)
- **Outputs** `Y_0..Y_5`: pre-tanh actuator commands from the full policy network

The latent space is purely an internal implementation detail — specs are always obs → action.

**Verification procedure (Method B):**

1. Parse input bounds from canonical spec using `read_vnnlib_simple`
2. Build a trivially-unsatisfiable `Specification` in memory (encoder output ≤ −1, impossible since ReLU ≥ 0) — no temporary spec file written
3. Run nnenum on the encoder only → collect all output star sets
4. Convert each star to an IQ-Verify `StarSet`; call `stateset_to_qpoints` (bounding-box method) to find all reachable quantized latent cell centers
5. Evaluate `latent_ctrl(cell)` → pre-tanh actions for each cell
6. Check action violation conditions parsed from the same canonical spec

For N-dimensional latent the quantized grid is N-dimensional; cell count grows as O(R/q)^N where R is the latent range and q is `quant_step`.

```bash
# Run all specs (latent dim 1)
python compare_halfcheetah.py --all

# Single spec
python compare_halfcheetah.py --spec_id 1

# Different run dir (e.g. latent dim 2 after training)
python compare_halfcheetah.py --all --run_dir sac_sweep_runs/HalfCheetah-v4/latent2/seed0
```

---

## HalfCheetah-v4 Specifications

All specs use the same 17-input observation space (VecNormalize-normalized). Input boxes are computed from rollout data conditioned on a behavioral mask (e.g. "forward velocity above 1.0"): each observation dimension's bounds are set to the 2nd–98th or 10th–90th percentile of observed values in that regime. Tighter percentile ranges (p10/p90) give a smaller, more conservative input box; wider ranges (p2/p98) cover more of the tail behavior. Outputs are 6 pre-tanh joint torques.

### Observation layout

| Index | Joint / quantity |
|---|---|
| X_0 | torso height |
| X_1 | torso pitch |
| X_2–X_8 | joint angles (back thigh/shin/foot, front thigh/shin/foot, front foot) |
| X_9 | forward velocity |
| X_10–X_16 | joint angular velocities |

### Action layout (pre-tanh)

| Index | Joint |
|---|---|
| Y_0 | back hip |
| Y_1 | back knee |
| Y_2 | back ankle |
| Y_3 | front hip |
| Y_4 | front knee (shin) |
| Y_5 | front ankle |

### Spec definitions

| Spec | Behavioral regime | Input box | Violation | Expected result |
|---|---|---|---|---|
| spec_1 | Nominal balanced running | 10th–90th percentile box, xvel ∈ [0.67, 1.21], \|pitch\| ≤ 0.3 | Y_4 ≥ 3.5 (front knee extreme up) | **SAFE** |
| spec_2 | Nominal balanced running | same as spec_1 | Y_4 ≤ −3.5 (front knee extreme down) | **SAFE** |
| spec_3 | Pitch instability | 2nd–98th percentile box, pitch ≥ 0.8 (tumbling) | any Y_i ≥ 3.0 | **SAFE** |
| spec_4 | Tight nominal running | 10th–90th percentile box, xvel ∈ [0.67, 1.21], \|pitch\| ≤ 0.3 | Y_4 ≥ 0.5 (front knee applies torque) | **UNSAFE** |

**spec_1/2:** Prove the front knee never fully saturates (pre-tanh ±3.5) during normal gait. Observed pre-tanh range is [−1.98, +2.13], well inside the threshold.

**spec_3:** Prove no actuator exceeds pre-tanh 3.0 even during extreme pitch instability. Despite the latent reaching ~9.83 during tumbling, the policy stays in a moderate torque regime — holds across all 2076 reachable cells.

**spec_4:** The front knee actively cycles through significant positive torques during normal running (pre-tanh up to ~2.09). The violation Y_4 ≥ 0.5 is expected to hold and confirms the gait cycle is reachable — 74 of 132 cells violate it.

---

## Timing Results (latent dim = 1, quant_step = 0.005)

```
  Spec      Result     Stars     Cells   Encoder(s)   Quant(s)   Total(s)
  ────────  ────────  ──────  ────────   ──────────  ─────────  ─────────
  spec_1    safe         527       132        1.340      0.510      1.850
  spec_2    safe         413       132        1.131      0.395      1.526
  spec_3    safe         255      2076        1.113      0.481      1.594
  spec_4    unsafe       414       132        1.178      0.398      1.576
```

All four specs complete in **1.5–1.9 seconds**. Encoder nnenum dominates (~70% of runtime); quantization and latent_ctrl evaluation across all cells adds < 0.5s.

Running nnenum on the full network (14 layers, 1024 ReLUs in the latent controller alone) exceeds 60s on every spec — in practice 30+ minutes before timing out or erroring. The speedup is structural: the encoder has only 17 ReLU neurons to case-split on, while the latent controller's 1024 ReLUs are bypassed entirely by the quantized lookup.

---

## File Structure

```
.
├── train_custom_sb3.py          # SAC training + model export (Hopper / HalfCheetah)
├── train_latent_sweep.py        # HalfCheetah latent dim 2 and 3 training
├── compare_halfcheetah.py       # HalfCheetah-v4 verification (all specs)
├── compare_verification.py      # Hopper-v5 verification
├── eval_quantized_policy.py     # Clean vs quantized-latent policy performance
├── generate_halfcheetah_specs.py  # Generate specs from rollout data
├── specs/
│   ├── Hopper-v5/               # spec_1..4.vnnlib
│   └── HalfCheetah-v4/          # spec_1..4.vnnlib (obs → action format)
├── sac_sweep_runs/
│   ├── HalfCheetah-v4/arch0/seed0/    # latent dim 1
│   ├── HalfCheetah-v4/latent2/seed0/  # latent dim 2 (after train_latent_sweep.py)
│   └── HalfCheetah-v4/latent3/seed0/  # latent dim 3 (after train_latent_sweep.py)
├── nnenum_package/              # nnenum (piecewise-linear reachability)
└── nnenum_package/iq_verify/    # IQ-Verify (quantized reachability)
```

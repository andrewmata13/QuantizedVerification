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

Each controller uses the largest quant_step that keeps quantized return within ~5% of clean. Larger latent dims tolerate coarser grids.

| Architecture | Clean return | Quant step | Quantized return |
|---|---|---|---|
| `[16, 1, 512, 512]` | 6683 ± 98 | 0.02 | 6595 ± 103 |
| `[16, 2, 512, 512]` | 8705 ± 1644 | 0.1 | 8659 ± 94 |
| `[16, 3, 512, 512]` | 13522 ± 59 | 0.05 | 12990 ± 159 |

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

## Verification Results

Quant steps chosen as the largest value keeping quantized return within ~5% of clean. Cell count and timing scale as O((range/q)^N) where N is the latent dim.

Cell enumeration uses global min/max across all encoder output stars per latent dimension — O(S×D) instead of O(N log N) deduplication. This is a slight overapproximation relative to per-star bounding boxes but eliminates the sorting bottleneck entirely.

### Soundness and completeness

Safe results are always sound: if no cell in the enumerated grid violates the spec, no reachable state can violate it (the grid over-covers the reachable set). Unsafe results from the bounding-box overapproximation may be false positives — a reported violation might fall outside all reachable encoder output star sets.

The `--complete` flag adds a lazy LP membership check for every candidate violation: for each violating cell, the script solves a joint LP against each encoder output star to confirm the cell is genuinely reachable. Only violations that pass this filter are reported.

```bash
python compare_halfcheetah.py --all --complete
```

This filter is tractable when the violation count is small. For latent dim 3, where specs 2–4 can produce millions of candidate violations from the global bounding box, the LP filter becomes impractical. In that case, `--complete` is recommended only when the violation count after the grid check is known to be small (e.g., spec_1 latent3: 395 violations confirmed in 11s).

### latent dim = 1, quant_step = 0.02

```
  Spec      Result     Stars     Cells   Encoder(s)   Quant(s)   Total(s)
  ────────  ────────  ──────  ────────   ──────────  ─────────  ─────────
  spec_1    safe         234        32        1.140      0.472      1.612
  spec_2    safe         273        34        1.191      0.022      1.213
  spec_3    safe         728       544        1.496      0.069      1.565
  spec_4    unsafe       750        34        1.404      0.056      1.460
```

### latent dim = 2, quant_step = 0.1

```
  Spec      Result     Stars     Cells   Encoder(s)   Quant(s)   Total(s)
  ────────  ────────  ──────  ────────   ──────────  ─────────  ─────────
  spec_1    safe         105      4386        1.102      0.473      1.574
  spec_2    safe        1071      7030        1.592      0.145      1.737
  spec_3    unsafe       632     16848        1.555      0.144      1.700
  spec_4    unsafe       874      7215        1.667      0.186      1.853
```

### latent dim = 3, quant_step = 0.05

```
  Spec      Result     Stars      Cells   Encoder(s)   Quant(s)   Total(s)
  ────────  ────────  ──────  ---------   ──────────  ─────────  ─────────
  spec_1    unsafe       168    2141916        1.050      4.924      5.974
  spec_2    unsafe      1034    2345908        2.279      4.501      6.779
  spec_3    unsafe       574   18358200        3.536     69.570     73.106  ⚠
  spec_4    unsafe      1034    2345908        7.842      3.550     11.392
```

⚠ spec_3 for latent dim 3 generates 18M cells due to the large latent range during pitch-instability states. Down from 441s to 73s after replacing O(N log N) dedup with O(S×D) global min/max.

spec_1 latent3 with `--complete`: all 395 candidate violations confirmed reachable via LP membership check (10.9s total) — result is complete, not just sound.

Encoder nnenum dominates for latent dim 1 (~75% of runtime). For higher dims the quantized evaluation becomes the bottleneck as cell count grows cubically.

Running nnenum on the full network (14 layers, 1024 ReLUs in the latent controller alone) exceeds 60s on every spec — in practice 30+ minutes before timing out or erroring. The speedup is structural: the encoder has only 17 ReLU neurons to case-split on, while the latent controller's 1024 ReLUs are bypassed entirely by the quantized lookup.

---

## File Structure

```
.
├── train_custom_sb3.py          # SAC training + model export (Hopper / HalfCheetah)
├── train_latent_sweep.py        # HalfCheetah latent dim 2 and 3 training
├── compare_halfcheetah.py       # HalfCheetah-v4 verification (all specs, --complete flag)
├── compare_verification.py      # Hopper-v5 verification
├── eval_quantized_policy.py     # Clean vs quantized-latent policy performance
├── generate_halfcheetah_specs.py  # Generate safety specs from rollout data
├── gen_robustness_specs.py      # Generate local robustness specs (L-inf ball around obs)
├── gen_trajectory_specs.py      # Generate paired SAT/UNSAT specs from trajectory states
├── specs/
│   ├── Hopper-v5/               # spec_1..4.vnnlib
│   └── HalfCheetah-v4/          # spec_1..4.vnnlib, rob_spec_*.vnnlib, traj_spec_*.vnnlib
├── sac_sweep_runs/
│   ├── HalfCheetah-v4/arch0/seed0/    # latent dim 1
│   ├── HalfCheetah-v4/latent2/seed0/  # latent dim 2 (after train_latent_sweep.py)
│   └── HalfCheetah-v4/latent3/seed0/  # latent dim 3 (after train_latent_sweep.py)
├── nnenum_package/              # nnenum (piecewise-linear reachability)
└── nnenum_package/iq_verify/    # IQ-Verify (quantized reachability)
```

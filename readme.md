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

1. **Encoder** (obs → N-D latent): small network, verified with nnenum (fast, complete)
2. **Latent controller** (N-D latent → action): verified by enumerating all reachable quantized latent cells via GPU lookup — no neural network verification needed

The key insight is structural: the encoder has only ~17 ReLU neurons to case-split on, while the latent controller's 512+ neurons are bypassed entirely by the quantized lookup. This gives >1000× speedup over alpha-beta CROWN on the same specs.

### Architecture

```
Observation (17)
    → Linear(17, 16) → ReLU
    → Linear(16, N)  → ReLU      ← bottleneck (N-D latent)
    → Linear(N, 512) → ReLU
    → Linear(512, 512) → ReLU
    → Linear(512, 6)              ← pre-tanh actions
    → Tanh                        ← final actions (applied at runtime only)
```

For N=1: the encoder is 6 nnenum layers (17 ReLU neurons); the full network is 14 nnenum layers (17 + 1024 ReLU neurons). Running nnenum on the full network exceeds 30+ minutes on every spec and typically times out. Running on the encoder only takes ~1.5s.

---

## Training

### HalfCheetah-v4

```bash
# Latent dim 1 (arch [16, 1, 512, 512])
python train_custom_sb3.py

# Latent dims 2 and 3
python train_latent_sweep.py

# Standard [256, 256] baseline (no bottleneck)
python train_baseline.py --env HalfCheetah-v4
```

### Hopper-v5

```bash
# Latent dims 3 and 4 (arch [16, N, 512, 512])
python train_hopper_latent_sweep.py

# Standard [256, 256] baseline
python train_baseline.py --env Hopper-v5
```

Each run saves to `sac_sweep_runs/<env>/<run_name>/seed0/`:
- `model.zip` — full SB3 SAC model
- `encoder.onnx` — encoder network (ONNX, ReLU-only, no Tanh)
- `encoder_full.pth`, `latent_controller_full.pth` — PyTorch modules
- `train_vec_norm.pkl` — VecNormalize running statistics
- `full_network.onnx` — full obs→action network for alpha-beta CROWN

All training scripts support checkpoint/resume: if `checkpoints/` contains `.zip` files from a prior run, training resumes from the latest checkpoint.

### Policy Performance

#### HalfCheetah-v4 (3M steps)

| Architecture | Mean return | Notes |
|---|---|---|
| `[256, 256]` baseline | 14380 ± 53 | No bottleneck |
| `[16, 1, 512, 512]` latent1 | 6683 ± 98 | |
| `[16, 2, 512, 512]` latent2 | 8705 ± 1644 | |
| `[16, 3, 512, 512]` latent3 | 13522 ± 59 | |

#### Hopper-v5 (3M steps)

| Architecture | Mean return | Notes |
|---|---|---|
| `[256, 256]` baseline | — | training in progress |
| `[16, 1, 512, 512]` arch0 | 1058 ± 0.5 | dim=1 too restrictive for Hopper |
| `[16, 2, 512, 512]` latent2 | 999 ± 120 | dim=2 still struggles |
| `[16, 3, 512, 512]` latent3 | 3401 ± 3 | recovers well |
| `[16, 4, 512, 512]` latent4 | — | training in progress |

---

## Quantization

Quant steps are chosen as the largest value keeping quantized return within ~5% of clean:

| Architecture | Clean return | Quant step | Quantized return |
|---|---|---|---|
| HalfCheetah latent1 | 6683 ± 98 | 0.02 | 6595 ± 103 |
| HalfCheetah latent2 | 8705 ± 1644 | 0.1 | 8659 ± 94 |
| HalfCheetah latent3 | 13522 ± 59 | 0.05 | 12990 ± 159 |

---

## Verification

### Approach

Specs are written in VNN-LIB format with:
- **Inputs** `X_0..X_16`: VecNormalize-normalized observations
- **Outputs** `Y_0..Y_5`: pre-tanh actuator commands from the full policy network

**Verification procedure:**

1. Parse input bounds from VNN-LIB spec
2. Run nnenum on the encoder only → collect all output star sets
3. Convert each star to a bounding box; enumerate all reachable quantized latent cell centers
4. Evaluate `latent_ctrl(cell)` → pre-tanh actions for each cell (batched GPU)
5. Check action violation conditions from the spec

For N-dimensional latent the quantized grid is N-dimensional; cell count grows as O(R/q)^N.

### Soundness and Completeness

**Safe results are always sound:** if no cell in the enumerated grid violates the spec, no reachable quantized state can violate it — the grid over-covers the reachable set.

**Unsafe results** from the bounding-box overapproximation may be false positives. The `--complete` flag adds a lazy LP membership check: for each candidate violation cell it solves a joint LP against each encoder output star. The check **short-circuits at the first confirmed violation** — only one witness is needed.

```bash
python compare_halfcheetah.py --all --complete
```

---

## HalfCheetah-v4 Specifications

### Observation / Action layout

| Index | Observation | | Index | Pre-tanh action |
|---|---|---|---|---|
| X_0 | torso height | | Y_0 | back hip |
| X_1 | torso pitch | | Y_1 | back knee |
| X_2–X_8 | joint angles | | Y_2 | back ankle |
| X_9 | forward velocity | | Y_3 | front hip |
| X_10–X_16 | joint angular velocities | | Y_4 | front knee |
| | | | Y_5 | front ankle |

### Spec definitions

All 4 specs use the **same input box**: p20/p80 percentile bounds from baseline rollouts during nominal balanced running (xvel ∈ [0.5, 1.5], |pitch| ≤ 0.3, |height| ≤ 0.4). This is a wide enough box that alpha-beta CROWN requires real branch-and-bound work to prove any spec. Each spec checks a different output dimension's upper saturation bound.

Thresholds were set at PGD_max + 1.5 margin across all networks, then verified safe via PGD with 40 restarts before writing.

| Spec | Output checked | Threshold | Semantics |
|---|---|---|---|
| spec_1 | Y_4 (front knee) ≥ 5.07 | upper | front knee never fully saturates upward |
| spec_2 | Y_3 (front hip) ≥ 6.87 | upper | front hip never fully saturates upward |
| spec_3 | Y_5 (front ankle) ≥ 6.03 | upper | front ankle never fully saturates upward |
| spec_4 | Y_1 (back knee) ≥ 6.37 | upper | back knee never fully saturates upward |

All 4 specs are **SAFE** for all networks: baseline [256,256] and bottleneck latent1/2/3.

**Note on spec generation:** Earlier versions used narrower boxes (p10/p90) and different thresholds calibrated only on bottleneck networks. Those were unsafe for the baseline. The current specs (generated by `generate_halfcheetah_specs_v2.py`) use p20/p80 and are universally safe, enabling a fair timing comparison with alpha-beta CROWN.

**Note on VNN-LIB novelty:** No public VNN-LIB specs for any MuJoCo continuous-control environment exist in VNN-COMP or the academic literature (only CartPole, LunarLander, and Dubins Rejoin have appeared). These are believed to be the first HalfCheetah VNN-LIB specs.

---

## Verification Results (HalfCheetah-v4, new specs)

All specs verified with `--complete`. Stars = nnenum output star sets from encoder; Cells = quantized latent cells evaluated.

### latent1 (dim=1, quant_step=0.02)

```
  Spec      Result   Stars   Cells   Total(s)
  ────────  ──────   ─────   ─────   ────────
  spec_1    safe        66      33     1.26s
  spec_2    safe       225      40     1.41s
  spec_3    safe       462      40     1.39s
  spec_4    safe       462      40     1.39s
```

### latent2 (dim=2, quant_step=0.1)

```
  Spec      Result   Stars   Cells   Total(s)
  ────────  ──────   ─────   ─────   ────────
  spec_1    safe       466     462    1.50s
  spec_2    safe       466     462    1.41s
  spec_3    safe       466     462    1.54s
  spec_4    safe       466     462    1.47s
```

### latent3 (dim=3, quant_step=0.05)

```
  Spec      Result   Stars     Cells   Total(s)
  ────────  ──────   ─────   ───────   ────────
  spec_1    safe       113   164,640    1.65s
  spec_2    safe       113   164,640    1.65s
  spec_3    safe       113   164,640    1.75s
  spec_4    safe       113   164,640    1.83s
```

Cell count grows cubically with latent dim: 33–40 cells (dim=1) → 462 (dim=2) → 164,640 (dim=3), yet verification time stays flat at ~1.5s because the GPU-batched lookup scales well.

---

## Alpha-Beta CROWN Comparison

alpha-beta CROWN is a state-of-the-art neural network verifier using bound propagation + branch-and-bound. It verifies the full continuous network (no quantization). We compare it against our method on the same 4 specs × 3 bottleneck controllers = 12 cases, plus the baseline as reference.

```bash
# Export full network ONNX first
python export_full_onnx.py

# Run comparison (1800s timeout per spec)
python compare_abcrown.py --env HalfCheetah-v4 --spec_type safety --timeout 1800
```

### Results

| Network | Spec | α-β CROWN | Ours | Speedup |
|---|---|---|---|---|
| baseline [256,256] | spec_1–4 | **timeout >1800s** | N/A (no bottleneck) | — |
| latent1 | spec_1 | **timeout >1800s** | safe 1.60s | **>1125×** |
| latent1 | spec_2 | **timeout >1800s** | safe 1.48s | **>1216×** |
| latent1 | spec_3 | **timeout >1800s** | safe 1.49s | **>1208×** |
| latent1 | spec_4 | **timeout >1800s** | safe 1.31s | **>1374×** |
| latent2 | spec_1 | **timeout >1800s** | safe 1.44s | **>1250×** |
| latent2 | spec_2 | **timeout >1800s** | safe 1.53s | **>1176×** |
| latent2 | spec_3 | **timeout >1800s** | safe 1.35s | **>1333×** |
| latent2 | spec_4 | **timeout >1800s** | safe 1.40s | **>1286×** |
| latent3 | spec_1 | **timeout >1800s** | safe 1.64s | **>1098×** |
| latent3 | spec_2 | **timeout >1800s** | safe 1.68s | **>1071×** |
| latent3 | spec_3 | **timeout >1800s** | safe 1.57s | **>1146×** |
| latent3 | spec_4 | **timeout >1800s** | safe 1.80s | **>1000×** |

**All 12 bottleneck cases: >1000× speedup.** The baseline network cannot be verified at all by alpha-beta CROWN within the timeout — demonstrating that the bottleneck structure is load-bearing for tractable verification, not just an architectural choice.

### Why the speedup

- **alpha-beta CROWN** must reason about the full obs→action mapping (17→16→N→512→512→6). For a 256×256 network with a large 17D input box, BaB generates millions of subdomains and still can't tighten the bounds enough to prove the spec within 30 minutes.
- **Our method** splits the problem: nnenum on the tiny encoder (17 ReLU neurons) → ~1.5s. The latent controller's 1024 ReLU neurons are never analyzed — they're evaluated by lookup.
- The bottleneck architecture is what makes our decomposition possible. The baseline [256,256] network has no such split point.

Note: alpha-beta CROWN verifies the continuous policy; our method verifies the quantized policy (what actually runs at deployment). Both guarantees are valid for their respective runtime policies.

---

## Trajectory Robustness Specs

Generated by `gen_trajectory_specs.py`: for each of 5 evenly-spaced states from a rollout, an L-inf ball of radius `eps=0.1` is drawn around the reference observation. Two specs are produced per state:

- **unsafe**: `delta = max_dev - 0.05` — a violation is witnessed by sampling 200 random points in the box
- **safe**: `delta` is increased until the verifier confirms no cell in the grid violates

```bash
python gen_trajectory_specs.py --env HalfCheetah-v4 \
    --run_dir sac_sweep_runs/HalfCheetah-v4/latent2/seed0 \
    --quant_step 0.1 --label latent2

python compare_halfcheetah.py \
    --spec_path specs/HalfCheetah-v4/traj_spec_*_latent2_*.vnnlib \
    --run_dir sac_sweep_runs/HalfCheetah-v4/latent2/seed0 --quant_step 0.1
```

### Results (HalfCheetah-v4, 5 states × 2 specs × 3 controllers = 30 specs)

All 30 specs verify correctly. Cell counts stay small because the L-inf obs ball maps to a narrow latent region.

| Controller | Stars | Cells | Total(s) |
|---|---|---|---|
| latent1 (dim=1) | 3–6 | 5–7 | ~1.0s |
| latent2 (dim=2) | 10–27 | 30–56 | ~1.3s |
| latent3 (dim=3) | 1–28 | 1440–16864 | ~1.3s |

---

## File Structure

```
.
├── train_custom_sb3.py              # SAC training for HalfCheetah latent dim 1
├── train_latent_sweep.py            # HalfCheetah latent dim 2 and 3
├── train_hopper_latent_sweep.py     # Hopper-v5 latent dim 3 and 4
├── train_baseline.py                # Standard [256,256] SAC baseline (both envs)
├── compare_halfcheetah.py           # HalfCheetah verification (--all, --complete)
├── compare_verification.py          # Hopper-v5 verification
├── compare_abcrown.py               # alpha-beta CROWN timing comparison
├── export_full_onnx.py              # Export full_network.onnx for alpha-beta CROWN
├── generate_halfcheetah_specs_v2.py # Generate safety specs (p20/p80, universally safe)
├── gen_robustness_specs.py          # L-inf robustness specs around reference states
├── gen_trajectory_specs.py          # Paired SAT/UNSAT specs from trajectory states
├── specs/
│   ├── HalfCheetah-v4/
│   │   ├── spec_{1..4}.vnnlib          # Main safety specs (p20/p80 nominal running)
│   │   ├── rob_spec_*.vnnlib           # Local robustness specs
│   │   └── traj_spec_*.vnnlib          # Trajectory robustness specs
│   └── Hopper-v5/
│       └── spec_{1..4}.vnnlib
├── sac_sweep_runs/
│   ├── HalfCheetah-v4/
│   │   ├── arch0/seed0/               # latent dim 1
│   │   ├── latent2/seed0/             # latent dim 2
│   │   ├── latent3/seed0/             # latent dim 3
│   │   └── baseline/seed0/            # standard [256,256]
│   └── Hopper-v5/
│       ├── arch0/seed0/               # latent dim 1
│       ├── latent2/seed0/             # latent dim 2
│       ├── latent3/seed0/             # latent dim 3
│       ├── latent4/seed0/             # latent dim 4
│       └── baseline/seed0/            # standard [256,256]
├── nnenum_package/nnenum/             # piecewise-linear reachability
└── nnenum_package/iq_verify/          # IQ-Verify (quantized reachability)
```

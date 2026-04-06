"""
gen_robustness_specs.py

Generate open-loop robustness vnnlib specs for HalfCheetah-v4 bottleneck policies.

Robustness spec:
  Input box  : obs_norm_i ∈ [ref_i - eps, ref_i + eps]  (clipped to [-clip_obs, clip_obs])
  Violation  : ∃ j s.t. |Y_j - a_ref_j| > delta
             = disjunction: (Y_j >= a_ref_j + delta) OR (Y_j <= a_ref_j - delta)

Reference states are the midpoints of the existing spec_1..4 input boxes — the same
operating regions already used for safety verification.  For each latent controller,
the reference pre-tanh action is evaluated by running the *quantized* controller
(encoder ONNX → quantize → latent_ctrl) at the reference state.

Also prints the quantized-controller episode reward for each run_dir.

Usage:
    python gen_robustness_specs.py
    python gen_robustness_specs.py --eps 0.1 --delta 0.5 --quant_step 0.05
"""

import os, argparse, pickle
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import onnxruntime as ort
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from nnenum.vnnlib import read_vnnlib_simple

N_INPUTS  = 17
N_ACTIONS = 6

SPEC_PATHS = [
    "specs/HalfCheetah-v4/spec_1.vnnlib",
    "specs/HalfCheetah-v4/spec_2.vnnlib",
    "specs/HalfCheetah-v4/spec_3.vnnlib",
    "specs/HalfCheetah-v4/spec_4.vnnlib",
]

RUN_DIRS = {
    "latent1": "sac_sweep_runs/HalfCheetah-v4/arch0/seed0",
    "latent2": "sac_sweep_runs/HalfCheetah-v4/latent2/seed0",
    "latent3": "sac_sweep_runs/HalfCheetah-v4/latent3/seed0",
}


def load_encoder(run_dir):
    enc_onnx = os.path.join(run_dir, "encoder.onnx")
    sess = ort.InferenceSession(enc_onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    n_latent = sess.get_outputs()[0].shape[0]
    return sess, input_name, n_latent


def quantized_action(obs_norm, enc_sess, enc_input_name, latent_ctrl, quant_step, device):
    x = obs_norm.astype(np.float32).flatten()   # rank-1 input
    z = enc_sess.run(None, {enc_input_name: x})[0].flatten()  # (n_latent,)
    z_q = (np.floor(z / quant_step) * quant_step + quant_step / 2.0).astype(np.float32)
    with torch.no_grad():
        z_t = torch.tensor(z_q, dtype=torch.float32, device=device).unsqueeze(0)
        pre_tanh = latent_ctrl(z_t).cpu().numpy()[0]
    return pre_tanh, z_q


def eval_quantized_reward(run_dir, quant_step, n_steps=1000, seed=42, device="cpu"):
    """Run one episode with the quantized controller, return total reward."""
    enc_sess, enc_input_name, n_latent = load_encoder(run_dir)
    ctrl_path = os.path.join(run_dir, "latent_controller_full.pth")
    latent_ctrl = torch.load(ctrl_path, weights_only=False).to(device).eval()

    with open(os.path.join(run_dir, "train_vec_norm.pkl"), "rb") as f:
        vec_norm = pickle.load(f)
    vec_norm.training = False

    env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
    eval_vn = VecNormalize(env, training=False, norm_obs=True, norm_reward=False,
                           clip_obs=vec_norm.clip_obs)
    eval_vn.obs_rms = vec_norm.obs_rms

    # VecNormalize with norm_obs=True already returns normalized obs from reset/step
    obs_norm = eval_vn.reset()[0]   # (17,)
    total_reward = 0.0

    for _ in range(n_steps):
        pre_tanh, _ = quantized_action(obs_norm, enc_sess, enc_input_name,
                                       latent_ctrl, quant_step, device)
        env_action = np.tanh(pre_tanh).reshape(1, -1)
        obs_norm, reward, done, _ = eval_vn.step(env_action)
        obs_norm = obs_norm[0]
        total_reward += float(reward[0])
        if done[0]:
            break

    eval_vn.close()
    return total_reward


def spec_midpoint(spec_path):
    """Return the midpoint (in normalized obs space) of a vnnlib input box."""
    vnnlib_data = read_vnnlib_simple(spec_path, N_INPUTS, N_ACTIONS)
    box, _ = vnnlib_data[0]
    box = np.array(box)          # (N_INPUTS, 2)
    return (box[:, 0] + box[:, 1]) / 2.0


def write_vnnlib(path, obs_ref, a_ref, eps, delta, clip_obs, label):
    lines = []
    lines.append(f"; Robustness spec — {label}")
    lines.append(f"; Input box: reference obs ± eps={eps}, clipped to ±{clip_obs}")
    lines.append(f"; Violation: any pre-tanh action deviates from reference by > delta={delta}")
    lines.append(f"; Reference pre-tanh actions: {[f'{v:.4f}' for v in a_ref]}")
    lines.append("")

    for i in range(N_INPUTS):
        lines.append(f"(declare-const X_{i}  Real)")
    for j in range(N_ACTIONS):
        lines.append(f"(declare-const Y_{j}  Real)")
    lines.append("")

    for i in range(N_INPUTS):
        lo = float(np.clip(obs_ref[i] - eps, -clip_obs, clip_obs))
        hi = float(np.clip(obs_ref[i] + eps, -clip_obs, clip_obs))
        lines.append(f"(assert (>= X_{i}  {lo:+.6f}))")
        lines.append(f"(assert (<= X_{i}  {hi:+.6f}))")
    lines.append("")

    lines.append(f"; Violation: |Y_j - a_ref_j| > delta={delta} for any j")
    lines.append("(assert (or")
    for j in range(N_ACTIONS):
        lines.append(f"    (and (>= Y_{j} {a_ref[j] + delta:+.6f}))")
        lines.append(f"    (and (<= Y_{j} {a_ref[j] - delta:+.6f}))")
    lines.append("))")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps",        type=float, default=0.1,
                    help="Input perturbation radius in normalized obs space (default 0.1)")
    ap.add_argument("--delta",      type=float, default=0.5,
                    help="Max allowed pre-tanh action deviation (default 0.5)")
    ap.add_argument("--quant_step", type=float, default=0.05)
    ap.add_argument("--n_steps",    type=int,   default=1000,
                    help="Episode length for reward evaluation")
    ap.add_argument("--skip_reward_eval", action="store_true",
                    help="Skip the episode reward evaluation (faster)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "specs/HalfCheetah-v4"

    # Compute reference states: midpoints of spec_1..4 input boxes
    ref_states = {}
    for sp in SPEC_PATHS:
        sid = os.path.basename(sp).replace(".vnnlib", "")   # spec_4 ... spec_7
        ref_states[sid] = spec_midpoint(sp)

    print(f"Reference states: midpoints of {list(ref_states.keys())}")
    print(f"eps={args.eps}  delta={args.delta}  quant_step={args.quant_step}\n")

    # Get clip_obs from any VecNormalize (same for all runs)
    with open(os.path.join(list(RUN_DIRS.values())[0], "train_vec_norm.pkl"), "rb") as f:
        vn = pickle.load(f)
    clip_obs = vn.clip_obs

    for run_label, run_dir in RUN_DIRS.items():
        enc_sess, enc_input_name, n_latent = load_encoder(run_dir)
        ctrl_path = os.path.join(run_dir, "latent_controller_full.pth")
        latent_ctrl = torch.load(ctrl_path, weights_only=False).to(device).eval()

        if not args.skip_reward_eval:
            reward = eval_quantized_reward(run_dir, args.quant_step,
                                           args.n_steps, device=device)
            print(f"=== {run_label}  (n_latent={n_latent}, quantized reward={reward:.1f}) ===")
        else:
            print(f"=== {run_label}  (n_latent={n_latent}) ===")

        for sid, obs_ref in ref_states.items():
            a_ref, z_q = quantized_action(obs_ref, enc_sess, enc_input_name,
                                          latent_ctrl, args.quant_step, device)

            spec_fname = f"rob_{sid}_{run_label}.vnnlib"
            spec_path  = os.path.join(out_dir, spec_fname)
            label = f"{run_label}, reference={sid} midpoint, quant_step={args.quant_step}"
            write_vnnlib(spec_path, obs_ref, a_ref, args.eps, args.delta, clip_obs, label)

            print(f"  {spec_fname}  z_q={np.round(z_q,3).tolist()}  "
                  f"a_ref=[{', '.join(f'{v:.3f}' for v in a_ref)}]")

        print()

    print(f"Done. Written to {out_dir}/rob_spec_*.vnnlib")


if __name__ == "__main__":
    main()

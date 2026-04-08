"""
gen_trajectory_specs.py

Generate local robustness VNN-LIB specs from trajectory states.

For each sampled state from a rollout:
  Input box  : L-inf ball around the reference observation: [obs - eps, obs + eps]
  Reference  : quantized-controller pre-tanh action at obs
  Violation  : any pre-tanh action component deviates from reference by > delta
             = (Y_j >= a_ref_j + delta) OR (Y_j <= a_ref_j - delta)

For each state the script finds two specs:
  - SAT  (unsafe): delta small enough that some state in the box violates it
                   — found by evaluating nearby quantized cells
  - SAFE (unsat) : delta increased until verification returns UNSAT

Works with any gym environment that has a trained bottleneck policy
(encoder.onnx + latent_controller_full.pth + train_vec_norm.pkl).

Usage:
    python gen_trajectory_specs.py --env HalfCheetah-v4 --quant_step 0.02
    python gen_trajectory_specs.py --env Hopper-v5 --run_dir sac_sweep_runs/Hopper-v5/arch0/seed0
    python gen_trajectory_specs.py --env HalfCheetah-v4 --n_states 10 --eps 0.1
    python gen_trajectory_specs.py --env HalfCheetah-v4 --delta_safe_steps 5 --delta_scale 2.0
"""

import os, argparse, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import onnxruntime as ort
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from nnenum.nnenum import set_exact_settings
from nnenum.settings import Settings
from nnenum.enumerate import enumerate_network
from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.specification import Specification, DisjunctiveSpec
from nnenum.vnnlib import get_num_inputs_outputs


# ── Environment metadata ──────────────────────────────────────────────────────

ENV_META = {
    "HalfCheetah-v4": dict(n_obs=17, n_act=6),
    "Hopper-v5":      dict(n_obs=11, n_act=3),
    "Walker2d-v5":    dict(n_obs=17, n_act=6),
    "Ant-v5":         dict(n_obs=27, n_act=8),
}


# ── Policy helpers ────────────────────────────────────────────────────────────

def load_encoder(run_dir):
    path = os.path.join(run_dir, "encoder.onnx")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return sess, input_name


def load_latent_ctrl(run_dir, device="cpu"):
    path = os.path.join(run_dir, "latent_controller_full.pth")
    return torch.load(path, weights_only=False).to(device).eval()


def load_vec_norm(run_dir, env_id):
    vn_path = os.path.join(run_dir, "train_vec_norm.pkl")
    env = DummyVecEnv([lambda: gym.make(env_id)])
    eval_env = VecNormalize.load(vn_path, env)
    eval_env.training = False
    eval_env.norm_reward = False
    return eval_env


def quantized_action(obs_norm, enc_sess, enc_input_name, latent_ctrl, quant_step, device):
    """Run encoder → quantize → latent_ctrl, return pre-tanh action."""
    x = obs_norm.astype(np.float32).flatten()
    z = enc_sess.run(None, {enc_input_name: x})[0].flatten()
    z_q = (np.floor(z / quant_step) * quant_step + quant_step / 2).astype(np.float32)
    with torch.no_grad():
        z_t = torch.tensor(z_q, dtype=torch.float32, device=device).unsqueeze(0)
        pre_tanh = latent_ctrl(z_t).cpu().numpy()[0]
    return pre_tanh


# ── Trajectory collection ─────────────────────────────────────────────────────

def collect_trajectory(eval_env, enc_sess, enc_input_name, latent_ctrl, quant_step,
                        n_steps=500, device="cpu"):
    """
    Roll out the quantized policy and collect (obs_norm, pre_tanh_action) pairs.
    Returns arrays of shape (T, n_obs) and (T, n_act).
    """
    obs_list, act_list = [], []
    obs = eval_env.reset()[0]

    for _ in range(n_steps):
        pre_tanh = quantized_action(obs, enc_sess, enc_input_name,
                                    latent_ctrl, quant_step, device)
        obs_list.append(obs.copy())
        act_list.append(pre_tanh.copy())

        env_action = np.tanh(pre_tanh).reshape(1, -1)
        result = eval_env.step(env_action)
        obs = result[0][0]
        done = bool(result[2][0] or result[3][0]) if len(result) == 4 else bool(result[2][0])
        if done:
            obs = eval_env.reset()[0]

    return np.array(obs_list), np.array(act_list)


# ── Verification ──────────────────────────────────────────────────────────────

def nnenum_config():
    set_exact_settings()
    Settings.RESULT_SAVE_STARS = True
    Settings.OVERAPPROX_BOTH_BOUNDS = True
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX


def verify_spec(encoder_onnx, latent_ctrl_path, obs_lo, obs_hi, a_ref, delta,
                quant_step, n_latent, inp_dtype, device):
    """
    Run encoder nnenum + quantized lookup for a single (box, delta) spec.
    Returns 'safe' or 'unsafe'.
    """
    init_box = np.stack([obs_lo, obs_hi], axis=1).astype(inp_dtype)
    trivial_spec = Specification(np.eye(n_latent), np.full(n_latent, -1.0))

    encoder_network = load_onnx_network_optimized(encoder_onnx)
    res = enumerate_network(init_box, encoder_network, trivial_spec)
    enc_stars = res.stars

    all_pts = []
    for s in enc_stars:
        star_lo = np.array([s.minimize_output(d, maximize=False) for d in range(n_latent)])
        star_hi = np.array([s.minimize_output(d, maximize=True)  for d in range(n_latent)])
        axes = []
        for d in range(n_latent):
            first = np.floor(star_lo[d] / quant_step) * quant_step + quant_step / 2
            last  = np.floor(star_hi[d] / quant_step) * quant_step + quant_step / 2
            axes.append(np.arange(first, last + quant_step * 0.5, quant_step))
        grids = np.meshgrid(*axes, indexing='ij')
        pts = np.stack([g.ravel() for g in grids], axis=1)
        all_pts.append(pts)

    if not all_pts:
        return "safe"

    all_pts = np.unique(np.round(np.concatenate(all_pts, axis=0), 6), axis=0)

    latent_ctrl = torch.load(latent_ctrl_path, weights_only=False).to(device).eval()
    chunk_size = 100_000
    actions_parts = []
    with torch.no_grad():
        for i in range(0, len(all_pts), chunk_size):
            z_chunk = torch.tensor(all_pts[i:i+chunk_size], dtype=torch.float32).to(device)
            actions_parts.append(latent_ctrl(z_chunk).cpu().numpy())
    actions = np.concatenate(actions_parts, axis=0)

    # Check violation: |Y_j - a_ref_j| > delta for any j
    violated = np.any(np.abs(actions - a_ref) > delta, axis=1)
    return "unsafe" if violated.any() else "safe"


# ── Spec writing ──────────────────────────────────────────────────────────────

def write_vnnlib(path, obs_lo, obs_hi, a_ref, delta, result, label, n_obs, n_act, eps):
    sat_note = "SAT — violation witnessed within the L-inf box." if result == "unsafe" \
               else "UNSAT — verified safe across all reachable cells."
    lines = [
        f"; Trajectory robustness spec — {label}",
        f"; Input box : L-inf ball of radius eps={eps:.4f} around reference obs",
        f"; Reference : quantized-controller pre-tanh action at reference obs",
        f"; Violation : |Y_j - a_ref_j| > delta={delta:.4f} for some j",
        f"; {sat_note}",
        "",
    ]
    for i in range(n_obs):
        lines.append(f"(declare-const X_{i}  Real)")
    for j in range(n_act):
        lines.append(f"(declare-const Y_{j}  Real)")
    lines.append("")
    for i in range(n_obs):
        lines.append(f"(assert (>= X_{i}  {obs_lo[i]:+.6f}))")
        lines.append(f"(assert (<= X_{i}  {obs_hi[i]:+.6f}))")
    lines.append("")
    lines.append(f"; Violation: |Y_j - a_ref_j| > delta={delta:.4f} for any j")
    lines.append("(assert (or")
    for j in range(n_act):
        lines.append(f"    (and (>= Y_{j} {a_ref[j] + delta:+.6f}))")
        lines.append(f"    (and (<= Y_{j} {a_ref[j] - delta:+.6f}))")
    lines.append("))")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env",           default="HalfCheetah-v4",
                    choices=list(ENV_META.keys()))
    ap.add_argument("--run_dir",       default=None)
    ap.add_argument("--n_states",      type=int,   default=10,
                    help="Number of reference states to sample from the trajectory")
    ap.add_argument("--n_steps",       type=int,   default=500,
                    help="Total rollout length to sample states from")
    ap.add_argument("--eps",           type=float, default=0.1,
                    help="L-inf radius around each reference observation")
    ap.add_argument("--quant_step",    type=float, default=0.005)
    ap.add_argument("--delta_margin",  type=float, default=0.05,
                    help="Subtract from max_dev to get SAT delta; add to get initial SAFE candidate")
    ap.add_argument("--delta_scale",   type=float, default=1.5,
                    help="Multiply delta by this each step when searching for SAFE")
    ap.add_argument("--delta_safe_steps", type=int, default=6,
                    help="Max delta-scaling steps to find a SAFE spec per state")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--label",         type=str,   default=None,
                    help="Controller label appended to output filenames, e.g. 'latent2'. "
                         "Produces traj_spec_{i}_{label}_unsafe.vnnlib. "
                         "Defaults to the last two path components of run_dir.")
    args = ap.parse_args()

    env_id  = args.env
    run_dir = args.run_dir or f"sac_sweep_runs/{env_id}/arch0/seed0"
    label   = args.label or "_".join(run_dir.rstrip("/").split("/")[-2:])
    meta    = ENV_META[env_id]
    n_obs, n_act = meta["n_obs"], meta["n_act"]
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = f"specs/{env_id}"

    encoder_onnx    = os.path.join(run_dir, "encoder.onnx")
    latent_ctrl_path = os.path.join(run_dir, "latent_controller_full.pth")
    _, n_latent, inp_dtype = get_num_inputs_outputs(encoder_onnx)

    print(f"env={env_id}  run_dir={run_dir}  label={label}")
    print(f"n_states={args.n_states}  eps={args.eps}  "
          f"quant_step={args.quant_step}  n_latent={n_latent}\n")

    nnenum_config()

    enc_sess, enc_input_name = load_encoder(run_dir)
    latent_ctrl = load_latent_ctrl(run_dir, device)
    eval_env = load_vec_norm(run_dir, env_id)

    print("Collecting trajectory ...")
    obs_traj, act_traj = collect_trajectory(
        eval_env, enc_sess, enc_input_name, latent_ctrl,
        args.quant_step, n_steps=args.n_steps, device=device,
    )
    eval_env.close()
    print(f"  Collected {len(obs_traj)} steps\n")

    # Evenly space reference states across the trajectory
    indices = np.linspace(0, len(obs_traj) - 1, args.n_states, dtype=int)

    written = []

    for idx, t in enumerate(indices):
        obs_ref = obs_traj[t]
        a_ref   = act_traj[t]
        obs_lo  = np.clip(obs_ref - args.eps, -10, 10)
        obs_hi  = np.clip(obs_ref + args.eps,  -10, 10)

        # Evaluate policy at box corners via quantized controller to find max deviation
        # Sample a grid of points in the box to estimate max_dev
        rng = np.random.default_rng(args.seed + idx)
        samples = rng.uniform(obs_lo, obs_hi, size=(200, n_obs)).astype(np.float32)
        sample_acts = np.array([
            quantized_action(s, enc_sess, enc_input_name, latent_ctrl, args.quant_step, device)
            for s in samples
        ])
        max_dev = np.abs(sample_acts - a_ref).max()

        print(f"State {idx+1} (step {t})  max_dev={max_dev:.4f}")

        # ── SAT spec ──────────────────────────────────────────────────────────
        delta_sat = max(max_dev - args.delta_margin, 0.01)
        fname_sat = f"traj_spec_{idx+1}_{label}_unsafe.vnnlib"
        spec_label = f"{env_id} {label} state {idx+1} step {t} eps={args.eps}"
        write_vnnlib(os.path.join(out_dir, fname_sat),
                     obs_lo, obs_hi, a_ref, delta_sat, "unsafe", spec_label, n_obs, n_act, args.eps)
        print(f"  unsafe: delta={delta_sat:.4f}  →  {fname_sat}")
        written.append((fname_sat, "unsafe", delta_sat))

        # ── SAFE spec: increase delta until verification returns safe ─────────
        delta_safe = max_dev + args.delta_margin
        found_safe = False
        for _ in range(args.delta_safe_steps):
            t0 = time.time()
            result = verify_spec(encoder_onnx, latent_ctrl_path,
                                 obs_lo, obs_hi, a_ref, delta_safe,
                                 args.quant_step, n_latent, inp_dtype, device)
            elapsed = time.time() - t0
            print(f"  safe search: delta={delta_safe:.4f}  →  {result}  ({elapsed:.2f}s)")
            if result == "safe":
                fname_safe = f"traj_spec_{idx+1}_{label}_safe.vnnlib"
                write_vnnlib(os.path.join(out_dir, fname_safe),
                             obs_lo, obs_hi, a_ref, delta_safe, "safe", spec_label, n_obs, n_act, args.eps)
                print(f"  safe:   delta={delta_safe:.4f}  →  {fname_safe}  (verified SAFE)")
                written.append((fname_safe, "safe", delta_safe))
                found_safe = True
                break
            delta_safe *= args.delta_scale

        if not found_safe:
            print(f"  safe search: no SAFE spec found within {args.delta_safe_steps} steps")
        print()

    print(f"{'File':<40}  {'Result':<8}  {'Delta':>8}")
    print("-" * 60)
    for fname, result, delta in written:
        print(f"  {fname:<38}  {result:<8}  {delta:>8.4f}")


if __name__ == "__main__":
    main()

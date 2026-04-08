"""
export_full_onnx.py

Export full-network ONNX (encoder + latent_controller concatenated, no Tanh)
for all trained bottleneck models, and for the baseline [256,256] model.

This is needed for alpha-beta CROWN verification, which verifies the full
end-to-end network against the VNN-LIB spec.

Outputs: <run_dir>/full_network.onnx for each run_dir.

Usage:
    python export_full_onnx.py
    python export_full_onnx.py --run_dirs sac_sweep_runs/HalfCheetah-v4/arch0/seed0
"""

import os, argparse
import torch
import torch.nn as nn

BOTTLENECK_RUN_DIRS = [
    "sac_sweep_runs/HalfCheetah-v4/arch0/seed0",
    "sac_sweep_runs/HalfCheetah-v4/latent2/seed0",
    "sac_sweep_runs/HalfCheetah-v4/latent3/seed0",
    "sac_sweep_runs/Hopper-v5/arch0/seed0",
]

BASELINE_RUN_DIRS = [
    "sac_sweep_runs/HalfCheetah-v4/baseline/seed0",
    "sac_sweep_runs/Hopper-v5/baseline/seed0",
]

# obs dims per env (inferred from encoder input, but listed here as fallback)
ENV_OBS_DIMS = {
    "HalfCheetah-v4": 17,
    "Hopper-v5": 11,
}


def obs_dim_from_path(run_dir):
    for env, dim in ENV_OBS_DIMS.items():
        if env in run_dir:
            return dim
    raise ValueError(f"Cannot infer obs_dim from {run_dir}")


def export_bottleneck(run_dir):
    enc_path  = os.path.join(run_dir, "encoder_full.pth")
    ctrl_path = os.path.join(run_dir, "latent_controller_full.pth")
    out_path  = os.path.join(run_dir, "full_network.onnx")

    if not os.path.exists(enc_path) or not os.path.exists(ctrl_path):
        print(f"  SKIP {run_dir} — encoder_full.pth or latent_controller_full.pth missing")
        return

    encoder    = torch.load(enc_path,  weights_only=False).cpu().eval()
    latent_ctrl = torch.load(ctrl_path, weights_only=False).cpu().eval()
    full_net   = nn.Sequential(encoder, latent_ctrl).eval()

    obs_dim = obs_dim_from_path(run_dir)
    dummy   = torch.randn(obs_dim)

    with torch.no_grad():
        out = full_net(dummy)
    n_act = out.shape[0]

    torch.onnx.export(
        full_net, dummy, out_path,
        input_names=["obs"], output_names=["pre_tanh_action"],
        opset_version=17,
        dynamo=False,
    )
    print(f"  OK  {out_path}  (obs({obs_dim}) -> action({n_act}))")


def export_baseline(run_dir):
    model_path = os.path.join(run_dir, "model.zip")
    out_path   = os.path.join(run_dir, "full_network.onnx")

    if not os.path.exists(model_path):
        print(f"  SKIP {run_dir} — model.zip missing (training not done yet?)")
        return

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium as gym

    # Determine env id
    env_id = None
    for e in ENV_OBS_DIMS:
        if e in run_dir:
            env_id = e
            break

    # Load model (env needed only to infer spaces; use a dummy)
    env = DummyVecEnv([lambda: gym.make(env_id)])
    vn_path = os.path.join(run_dir, "train_vec_norm.pkl")
    if os.path.exists(vn_path):
        env = VecNormalize.load(vn_path, env)
        env.training = False
    model = SAC.load(model_path, env=env)

    # Extract actor: latent_pi + mu (no Tanh)
    actor = model.actor
    full_net = nn.Sequential(actor.latent_pi, actor.mu).cpu().eval()

    obs_dim = ENV_OBS_DIMS[env_id]
    dummy   = torch.randn(obs_dim)
    with torch.no_grad():
        out = full_net(dummy)
    n_act = out.shape[0]

    torch.onnx.export(
        full_net, dummy, out_path,
        input_names=["obs"], output_names=["pre_tanh_action"],
        opset_version=17,
        dynamo=False,
    )
    print(f"  OK  {out_path}  (obs({obs_dim}) -> action({n_act}))")
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="*", default=None,
                    help="Specific run dirs to export. Default: all known dirs.")
    args = ap.parse_args()

    if args.run_dirs:
        dirs = args.run_dirs
        for d in dirs:
            if "baseline" in d:
                export_baseline(d)
            else:
                export_bottleneck(d)
        return

    print("=== Bottleneck networks ===")
    for d in BOTTLENECK_RUN_DIRS:
        export_bottleneck(d)

    print("\n=== Baseline networks ===")
    for d in BASELINE_RUN_DIRS:
        export_baseline(d)


if __name__ == "__main__":
    main()

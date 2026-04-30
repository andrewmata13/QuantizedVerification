"""
figures/latent_heatmaps.py

Visualize the 2D latent space of the HalfCheetah-v4 latent2 policy.

Two figure types:
  1. Per-action heatmaps  (--mode actions)
     6 subplots, one per pre-tanh action dimension. Shows how each joint
     torque varies across (z1, z2).

  2. Semantic heatmap  (--mode semantic)
     Single figure colored by dominant action, magnitude, or a custom
     projection. Overlay rollout trajectories if --traj_npy is provided.

Usage:
    python figures/latent_heatmaps.py                      # action heatmaps
    python figures/latent_heatmaps.py --mode semantic
    python figures/latent_heatmaps.py --traj_npy figures/latent_traj.npy

To collect rollout trajectories (requires MuJoCo):
    python figures/collect_latent_traj.py
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ── Config ─────────────────────────────────────────────────────────────────────

RUN_DIR    = "sac_sweep_runs/HalfCheetah-v4/latent2/seed0"
OUT_DIR    = "figures"
RESOLUTION = 400          # grid points per axis
Z_RANGE    = (0.0, 1.8)   # covers the full reachable latent region (ReLU >= 0)
QUANT_STEP = 0.1          # quantization step for grid overlay

ACTION_NAMES = [
    "back hip\n(Y_0)",
    "back knee\n(Y_1)",
    "back ankle\n(Y_2)",
    "front hip\n(Y_3)",
    "front knee\n(Y_4)",
    "front ankle\n(Y_5)",
]

CMAP_ACTION   = "RdBu_r"    # diverging: negative → blue, positive → red
CMAP_SEMANTIC = "tab10"


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_controller(run_dir):
    path = os.path.join(run_dir, "latent_controller_full.pth")
    ctrl = torch.load(path, map_location="cpu", weights_only=False)
    ctrl.eval()
    return ctrl


def eval_grid(ctrl, resolution, z_range):
    """Evaluate controller on a dense (resolution x resolution) grid.

    Returns:
        z1_vals, z2_vals  — 1-D arrays of grid coordinates
        actions           — (resolution, resolution, 6) pre-tanh actions
    """
    z1_vals = np.linspace(z_range[0], z_range[1], resolution, dtype=np.float32)
    z2_vals = np.linspace(z_range[0], z_range[1], resolution, dtype=np.float32)
    Z1, Z2 = np.meshgrid(z1_vals, z2_vals)          # both (res, res)
    flat = np.stack([Z1.ravel(), Z2.ravel()], axis=1)  # (res^2, 2)

    chunk = 50_000
    outs = []
    with torch.no_grad():
        for i in range(0, len(flat), chunk):
            z = torch.from_numpy(flat[i:i+chunk])
            outs.append(ctrl(z).numpy())
    actions_flat = np.concatenate(outs, axis=0)      # (res^2, 6)
    actions = actions_flat.reshape(resolution, resolution, 6)
    return z1_vals, z2_vals, actions


def quant_grid_lines(z_range, quant_step):
    """Cell boundaries for quantization grid overlay."""
    lo, hi = z_range
    lines = np.arange(lo, hi + quant_step, quant_step)
    return lines


# ── Figure 1: per-action heatmaps ─────────────────────────────────────────────

def plot_action_heatmaps(z1, z2, actions, quant_step, out_path, show_quant_grid=True):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()

    # Shared symmetric colorbar limits per action
    for i, ax in enumerate(axes):
        data = actions[:, :, i]
        vmax = np.abs(data).max()
        im = ax.imshow(
            data,
            origin="lower",
            extent=[z1[0], z1[-1], z2[0], z2[-1]],
            aspect="equal",
            cmap=CMAP_ACTION,
            vmin=-vmax,
            vmax=vmax,
            interpolation="bilinear",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="pre-tanh")

        if show_quant_grid:
            lines = quant_grid_lines((z1[0], z1[-1]), quant_step)
            for v in lines:
                ax.axvline(v, color="k", lw=0.3, alpha=0.35)
                ax.axhline(v, color="k", lw=0.3, alpha=0.35)

        ax.set_title(ACTION_NAMES[i], fontsize=10)
        ax.set_xlabel("z₁", fontsize=9)
        ax.set_ylabel("z₂", fontsize=9)

    fig.suptitle(
        "HalfCheetah-v4 latent2: pre-tanh action heatmaps over (z₁, z₂)",
        fontsize=12, y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 2: semantic heatmap ─────────────────────────────────────────────────

# K-means cluster index → (semantic label, color)
# Clusters discovered from k=4 on action vectors; labels assigned by inspecting
# cluster centers against HalfCheetah action anatomy.
#
# Cluster centers (Y0 back-hip, Y1 back-knee, Y2 back-ankle,
#                  Y3 front-hip, Y4 front-knee, Y5 front-ankle):
#   Peak push    Y0=+2.55 Y2=+2.75 Y1=-2.61 (back spring fully released)
#   Back drive   Y0=+2.03 Y2=+1.88 Y3=+1.01 (sustained propulsion)
#   Front reach  Y0=-1.29 Y3=+1.49 Y5=+0.96 (back recovering, front swinging)
#   Front landing Y0=-2.00 Y4=+1.76          (front catching, back coiling)
#
# Assigned after running fit; update CLUSTER_LABELS if re-running k-means.
CLUSTER_LABELS = {
    # cluster_idx: (display_name, hex_color)
    0: ("Peak push",     "#C0392B"),   # dark red   — maximum back extension
    1: ("Front reach",   "#2980B9"),   # blue       — front swinging forward
    2: ("Back drive",    "#E67E22"),   # orange     — sustained propulsion
    3: ("Front landing", "#27AE60"),   # green      — front catching, back coiling
}
N_CLUSTERS = len(CLUSTER_LABELS)


def fit_kmeans(actions_flat, n_clusters=N_CLUSTERS, seed=42):
    """Fit k-means on (N, 6) action vectors. Returns fitted KMeans object."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    km.fit(actions_flat)
    return km


def remap_clusters(km, cluster_labels=CLUSTER_LABELS):
    """Remap raw k-means cluster indices to semantic labels by matching cluster
    centers to the expected signatures (highest Y0+Y2 → Peak push, etc.)."""
    centers = km.cluster_centers_  # (k, 6)
    back_drive_score = centers[:, 0] + centers[:, 2]   # Y0 + Y2
    front_reach_score = centers[:, 3] - centers[:, 0]  # Y3 - Y0 (front up, back down)
    front_land_score  = centers[:, 4] - centers[:, 0]  # Y4 - Y0 (fknee up, back down)

    # Assign: peak push = highest back drive with most bent back knee
    back_knee_bent = -centers[:, 1]   # Y1 most negative → most bent
    peak_push_score = back_drive_score + 0.5 * back_knee_bent

    order = np.argsort(-peak_push_score)  # descending
    peak_push_raw  = order[0]
    back_drive_raw = order[1]
    # Among remaining two, highest front_reach vs front_land
    remaining = [i for i in range(N_CLUSTERS)
                 if i not in (peak_push_raw, back_drive_raw)]
    if front_reach_score[remaining[0]] > front_reach_score[remaining[1]]:
        front_reach_raw, front_land_raw = remaining[0], remaining[1]
    else:
        front_reach_raw, front_land_raw = remaining[1], remaining[0]

    # Build remapping: raw_idx → semantic_idx (0=peak push,1=front reach,2=back drive,3=front land)
    remap = {
        peak_push_raw:  0,
        back_drive_raw: 2,
        front_reach_raw: 1,
        front_land_raw: 3,
    }
    return remap


def plot_semantic_heatmap(z1, z2, actions, quant_step, out_path,
                          traj_z=None, show_quant_grid=True):
    """
    Left: behavioral mode map from k-means on action vectors (k=4), continuous.
    Right: quantized version — each grid cell colored by k-means cluster of its center.
    """
    from matplotlib.patches import Rectangle

    res = actions.shape[0]
    actions_flat = actions.reshape(-1, 6)

    km = fit_kmeans(actions_flat)
    remap = remap_clusters(km)

    raw_labels = km.labels_.reshape(res, res)
    labels = np.vectorize(remap.__getitem__)(raw_labels)

    color_list = [CLUSTER_LABELS[i][1] for i in range(N_CLUSTERS)]
    color_arr  = np.array([mcolors.to_rgb(c) for c in color_list])
    rgb = color_arr[labels]

    # Evaluate controller at quantized cell centers
    ctrl = load_controller(RUN_DIR)
    lo, hi = z1[0], z1[-1]
    half = quant_step / 2.0
    c_vals = np.arange(lo + half, hi, quant_step, dtype=np.float32)
    n_cells = len(c_vals)
    C1, C2 = np.meshgrid(c_vals, c_vals)
    flat_cells = np.stack([C1.ravel(), C2.ravel()], axis=1)

    with torch.no_grad():
        cell_actions = ctrl(torch.from_numpy(flat_cells)).numpy()
    cell_labels_raw = km.predict(cell_actions)
    cell_labels = np.vectorize(remap.__getitem__)(cell_labels_raw)
    cell_labels = cell_labels.reshape(n_cells, n_cells)

    AXIS_FS = 14
    TICK_FS = 12

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: continuous mode map ───────────────────────────────────────────
    ax = axes[0]
    ax.imshow(
        rgb, origin="lower",
        extent=[z1[0], z1[-1], z2[0], z2[-1]],
        aspect="equal", interpolation="nearest",
    )
    if show_quant_grid:
        lines = quant_grid_lines((z1[0], z1[-1]), quant_step)
        for v in lines:
            ax.axvline(v, color="w", lw=0.4, alpha=0.35)
            ax.axhline(v, color="w", lw=0.4, alpha=0.35)

    ax.set_xlabel("z₁", fontsize=AXIS_FS)
    ax.set_ylabel("z₂", fontsize=AXIS_FS)
    ax.tick_params(labelsize=TICK_FS)

    # ── Right: quantized cell grid ──────────────────────────────────────────
    ax2 = axes[1]

    cell_rgb = color_arr[cell_labels]
    grid_lo = lo
    grid_hi = lo + n_cells * quant_step
    ax2.imshow(
        cell_rgb, origin="lower",
        extent=[grid_lo, grid_hi, grid_lo, grid_hi],
        aspect="equal", interpolation="nearest",
    )

    for v in np.arange(grid_lo, grid_hi + quant_step * 0.5, quant_step):
        ax2.axvline(v, color="k", lw=0.5, alpha=0.4)
        ax2.axhline(v, color="k", lw=0.5, alpha=0.4)

    ax2.set_xlim(grid_lo, grid_hi)
    ax2.set_ylim(grid_lo, grid_hi)
    ax2.set_xlabel("z₁", fontsize=AXIS_FS)
    ax2.set_ylabel("z₂", fontsize=AXIS_FS)
    ax2.tick_params(labelsize=TICK_FS)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Quantized grid: {n_cells}×{n_cells} = {n_cells**2} cells")


# ── Figure 3: spec boundary overlay ───────────────────────────────────────────

SAFE_SPECS = [
    # (action_dim, threshold, short_label, color)
    (4, 5.0668,  "Spec 1: Y₄ (front knee)",   "#1E88E5"),
    (3, 6.8722,  "Spec 2: Y₃ (front hip)",    "#D32F2F"),
    (5, 6.0250,  "Spec 3: Y₅ (front ankle)",  "#43A047"),
    (1, 6.3690,  "Spec 4: Y₁ (back knee)",    "#7B1FA2"),
]


def _enumerate_encoder_stars(spec_path, encoder_onnx, n_lat):
    """Run nnenum on encoder, return list of output stars and their bounding box."""
    from nnenum.settings import Settings
    Settings.CHECK_SINGLE_THREAD_BLAS = False
    from nnenum.nnenum import set_exact_settings
    from nnenum.enumerate import enumerate_network
    from nnenum.onnx_network import load_onnx_network_optimized
    from nnenum.specification import Specification
    from nnenum.vnnlib import read_vnnlib_simple, get_num_inputs_outputs

    n_in, _, _ = get_num_inputs_outputs(encoder_onnx)

    set_exact_settings()
    Settings.CHECK_SINGLE_THREAD_BLAS = False
    Settings.RESULT_SAVE_STARS = True
    Settings.OVERAPPROX_BOTH_BOUNDS = True
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX

    net = load_onnx_network_optimized(encoder_onnx)
    vnnlib = read_vnnlib_simple(spec_path, n_in, n_lat + 6)
    box, _ = vnnlib[0]
    init_box = np.array(box, dtype=np.float32)

    trivial_spec = Specification(np.eye(n_lat), np.full(n_lat, -1.0))
    res = enumerate_network(init_box, net, trivial_spec)
    stars = res.stars

    global_lo = np.full(n_lat, np.inf)
    global_hi = np.full(n_lat, -np.inf)
    for s in stars:
        lo = np.array([s.minimize_output(d, maximize=False) for d in range(n_lat)])
        hi = np.array([s.minimize_output(d, maximize=True) for d in range(n_lat)])
        global_lo = np.minimum(global_lo, lo)
        global_hi = np.maximum(global_hi, hi)

    return stars, global_lo, global_hi


def plot_spec_overlay(z1, z2, actions, quant_step, out_path, show_quant_grid=True):
    """
    Left: full latent space with gait-phase coloring and quantization grid.
    Right: zoomed to reachable region from spec_5 (p32/p68), showing encoder
           output star sets and the quantized cells they cover.
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle, Polygon

    STYLE = os.path.join(os.path.dirname(__file__), "bak_matplotlib.mlpstyle")
    if os.path.exists(STYLE):
        plt.style.use(STYLE)

    res = actions.shape[0]
    actions_flat = actions.reshape(-1, 6)

    km = fit_kmeans(actions_flat)
    remap = remap_clusters(km)
    raw_labels = km.labels_.reshape(res, res)
    labels = np.vectorize(remap.__getitem__)(raw_labels)

    color_list = [CLUSTER_LABELS[i][1] for i in range(N_CLUSTERS)]
    color_arr  = np.array([mcolors.to_rgb(c) for c in color_list])
    rgb = color_arr[labels]

    # Run encoder enumeration for spec_5
    enc_onnx = os.path.join(RUN_DIR, "encoder.onnx")
    spec_path = "specs/HalfCheetah-v4/spec_5.vnnlib"
    print("Running encoder enumeration for spec_5 ...")
    stars, global_lo, global_hi = _enumerate_encoder_stars(spec_path, enc_onnx, 2)
    print(f"  {len(stars)} stars, bbox z1=[{global_lo[0]:.3f}, {global_hi[0]:.3f}], "
          f"z2=[{global_lo[1]:.3f}, {global_hi[1]:.3f}]")

    # Enumerate reachable cells from bounding box
    half = quant_step / 2.0
    axes_list = []
    for d in range(2):
        first = np.floor(global_lo[d] / quant_step) * quant_step + half
        last = np.floor(global_hi[d] / quant_step) * quant_step + half
        axes_list.append(np.arange(first, last + quant_step * 0.5, quant_step))
    C1, C2 = np.meshgrid(axes_list[0], axes_list[1])
    reachable_cells = np.stack([C1.ravel(), C2.ravel()], axis=1)

    # Extract star polygons (each star.verts() returns a list of 2D vertices)
    star_polys = []
    for s in stars:
        v = s.verts()
        if v is not None and len(v) >= 3:
            pts = np.array(v)
            star_polys.append(pts)

    print(f"  {len(star_polys)} plottable star polygons, "
          f"{len(reachable_cells)} reachable cells")

    fig, axes_arr = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: full latent space with gait phases ─────────────────────────────
    ax = axes_arr[0]
    ax.imshow(
        rgb, origin="lower",
        extent=[z1[0], z1[-1], z2[0], z2[-1]],
        aspect="equal", interpolation="nearest",
    )
    if show_quant_grid:
        lines = quant_grid_lines((z1[0], z1[-1]), quant_step)
        for v in lines:
            ax.axvline(v, color="w", lw=0.4, alpha=0.35)
            ax.axhline(v, color="w", lw=0.4, alpha=0.35)

    # Draw the reachable bounding box on the left panel
    bb_w = global_hi[0] - global_lo[0]
    bb_h = global_hi[1] - global_lo[1]
    reach_rect = Rectangle(
        (global_lo[0], global_lo[1]), bb_w, bb_h,
        linewidth=2.5, edgecolor="white", facecolor="none",
        linestyle="-", zorder=5,
    )
    ax.add_patch(reach_rect)
    ax.annotate(
        "Reachable region\n(spec 5, p32/p68)",
        xy=(bb_w / 2, global_hi[1]), xytext=(0.55, 1.35),
        fontsize=9, color="white", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
        zorder=6,
    )

    legend_patches = []
    for sem_idx in range(N_CLUSTERS):
        name, color = CLUSTER_LABELS[sem_idx]
        pct = (labels == sem_idx).mean() * 100
        legend_patches.append(mpatches.Patch(color=color, label=f"{name}  ({pct:.0f}%)"))
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.9,
              title="Gait phase", title_fontsize=8)
    ax.set_xlabel("z₁", fontsize=10)
    ax.set_ylabel("z₂", fontsize=10)
    ax.set_title("Full latent space", fontsize=11)

    # ── Right: zoomed view with stars and cells ──────────────────────────────
    ax2 = axes_arr[1]

    pad = quant_step * 1.0
    x_lo = min(global_lo[0], 0) - pad
    x_hi = global_hi[0] + pad
    y_lo = min(global_lo[1], 0) - pad
    y_hi = global_hi[1] + pad

    ax2.imshow(
        rgb, origin="lower",
        extent=[z1[0], z1[-1], z2[0], z2[-1]],
        aspect="equal", interpolation="nearest",
        alpha=0.4,
    )

    # Draw quantization grid
    for v in np.arange(0, x_hi + quant_step, quant_step):
        ax2.axvline(v, color="gray", lw=0.5, alpha=0.3)
    for v in np.arange(0, y_hi + quant_step, quant_step):
        ax2.axhline(v, color="gray", lw=0.5, alpha=0.3)

    # Draw each verified cell as a highlighted rectangle
    for c1, c2 in reachable_cells:
        rect = Rectangle(
            (c1 - half, c2 - half), quant_step, quant_step,
            linewidth=0.6, edgecolor="#333333", facecolor="#43A047",
            alpha=0.25, zorder=2,
        )
        ax2.add_patch(rect)

    # Draw star set polygons
    for pts in star_polys:
        poly = Polygon(
            pts, closed=True,
            facecolor="#1565C0", edgecolor="#0D47A1",
            alpha=0.35, linewidth=0.7, zorder=3,
        )
        ax2.add_patch(poly)

    # Mark cell centers
    ax2.scatter(
        reachable_cells[:, 0], reachable_cells[:, 1],
        s=15, c="white", edgecolors="k", linewidths=0.6,
        zorder=4, label=f"Verified cells ({len(reachable_cells)})",
    )

    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_xlabel("z₁", fontsize=10)
    ax2.set_ylabel("z₂", fontsize=10)
    ax2.set_title("Encoder stars → quantized cells (spec 5)", fontsize=11)

    # Legend for right panel
    star_patch = mpatches.Patch(facecolor="#1565C0", edgecolor="#0D47A1",
                                alpha=0.5, label=f"Encoder stars ({len(star_polys)})")
    cell_patch = mpatches.Patch(facecolor="#43A047", edgecolor="#333333",
                                alpha=0.4, label=f"Verified cells ({len(reachable_cells)})")
    ax2.legend(handles=[star_patch, cell_patch], loc="upper right",
               fontsize=8, framealpha=0.9)

    n_cells = len(reachable_cells)
    ax2.text(
        0.02, 0.97, "All SAFE",
        transform=ax2.transAxes, fontsize=10, va="top", fontweight="bold",
        color="#2E7D32",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#43A047", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Stars: {len(star_polys)}, cells: {n_cells}")


# ── Figure 4: quantized cell grid ─────────────────────────────────────────────

def plot_cell_grid(ctrl, quant_step, z_range, out_path):
    """
    Left: continuous gait-mode map (k-means on dense grid).
    Right: the actual 18×18 quantized cells, each colored by the most-binding
           spec utilization at that cell center.  Threshold = 100% (violation).
    """
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.ticker as ticker

    STYLE = os.path.join(os.path.dirname(__file__), "bak_matplotlib.mlpstyle")
    if os.path.exists(STYLE):
        plt.style.use(STYLE)

    # ── Dense grid for left panel ────────────────────────────────────────────
    z1, z2, actions = eval_grid(ctrl, RESOLUTION, z_range)
    res = actions.shape[0]
    actions_flat = actions.reshape(-1, 6)

    km = fit_kmeans(actions_flat)
    remap = remap_clusters(km)
    raw_labels = km.labels_.reshape(res, res)
    labels = np.vectorize(remap.__getitem__)(raw_labels)
    color_list = [CLUSTER_LABELS[i][1] for i in range(N_CLUSTERS)]
    color_arr  = np.array([mcolors.to_rgb(c) for c in color_list])
    rgb = color_arr[labels]

    # ── Cell centers for right panel ─────────────────────────────────────────
    lo, hi = z_range
    # cell centers: first = lo + q/2, step = q
    half = quant_step / 2.0
    c_vals = np.arange(lo + half, hi, quant_step, dtype=np.float32)
    n_cells = len(c_vals)

    C1, C2 = np.meshgrid(c_vals, c_vals)          # (n, n)
    flat_cells = np.stack([C1.ravel(), C2.ravel()], axis=1)

    with torch.no_grad():
        cell_actions = ctrl(torch.from_numpy(flat_cells)).numpy()  # (n², 6)
    cell_actions = cell_actions.reshape(n_cells, n_cells, 6)

    # Utilization per cell: max(Y_i / threshold_i) across all specs, clamped ≥ 0
    util = np.zeros((n_cells, n_cells))
    for dim, thr, _, _ in SAFE_SPECS:
        util = np.maximum(util, cell_actions[:, :, dim] / thr)
    util_pct = util * 100  # percentage

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: gait mode map ──────────────────────────────────────────────────
    ax = axes[0]
    ax.imshow(rgb, origin="lower",
              extent=[z1[0], z1[-1], z2[0], z2[-1]],
              aspect="equal", interpolation="nearest")
    lines = quant_grid_lines((z1[0], z1[-1]), quant_step)
    for v in lines:
        ax.axvline(v, color="w", lw=0.4, alpha=0.35)
        ax.axhline(v, color="w", lw=0.4, alpha=0.35)

    patches = [
        mpatches.Patch(color=CLUSTER_LABELS[i][1],
                       label=f"{CLUSTER_LABELS[i][0]}  "
                             f"({(labels == i).mean()*100:.0f}%)")
        for i in range(N_CLUSTERS)
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.9,
              title="Gait phase", title_fontsize=8)
    ax.set_xlabel("z₁", fontsize=10)
    ax.set_ylabel("z₂", fontsize=10)

    # ── Right: cell grid ─────────────────────────────────────────────────────
    ax2 = axes[1]

    # Traffic-light colormap: white (safe) → green → yellow → orange → red (violation)
    safe_cmap = LinearSegmentedColormap.from_list(
        "cell_safe",
        [(0.0, "#FFFFFF"), (0.3, "#43A047"), (0.55, "#FDD835"),
         (0.75, "#FB8C00"), (1.0, "#C62828")],
    )

    im = ax2.imshow(
        util_pct, origin="lower", aspect="equal",
        extent=[lo, hi, lo, hi],
        cmap=safe_cmap, vmin=0, vmax=100,
        interpolation="nearest",
    )

    # Cell boundary lines
    for v in np.arange(lo, hi + quant_step, quant_step):
        ax2.axvline(v, color="k", lw=0.5, alpha=0.35)
        ax2.axhline(v, color="k", lw=0.5, alpha=0.35)

    # Annotate max cell
    peak_idx = np.unravel_index(util_pct.argmax(), util_pct.shape)
    peak_c1 = c_vals[peak_idx[1]]
    peak_c2 = c_vals[peak_idx[0]]
    peak_val = util_pct[peak_idx]
    ax2.plot(peak_c1, peak_c2, "*", color="white", markersize=13,
             markeredgecolor="k", markeredgewidth=1.2, zorder=5)
    text_x = peak_c1 - 0.55 if peak_c1 > 1.0 else peak_c1 + 0.20
    ax2.annotate(
        f"max {peak_val:.0f}%",
        xy=(peak_c1, peak_c2),
        xytext=(text_x, peak_c2 + 0.22),
        fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="k", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", alpha=0.92),
        zorder=6,
    )

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Spec utilization (%)", fontsize=9)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(["0", "25", "50", "75", "100\n(violation)"])

    ax2.set_xlabel("z₁", fontsize=10)
    ax2.set_ylabel("z₂", fontsize=10)
    ax2.set_xlim(lo, hi - quant_step)
    ax2.set_ylim(lo, hi - quant_step)
    ax2.text(0.02, 0.97, f"{n_cells}×{n_cells} = {n_cells**2} cells verified",
             transform=ax2.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Cell grid: {n_cells}×{n_cells} = {n_cells**2} cells, "
          f"max utilization = {util_pct.max():.1f}%")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",       default="actions",
                    choices=["actions", "semantic", "specoverlay", "cells", "both"])
    ap.add_argument("--run_dir",    default=RUN_DIR)
    ap.add_argument("--out_dir",    default=OUT_DIR)
    ap.add_argument("--resolution", type=int, default=RESOLUTION)
    ap.add_argument("--z_range",    type=float, nargs=2, default=list(Z_RANGE))
    ap.add_argument("--quant_step", type=float, default=QUANT_STEP)
    ap.add_argument("--no_grid",    action="store_true", help="Hide quantization grid lines")
    ap.add_argument("--traj_npy",   default=None,
                    help="Path to .npy file with rollout latent coords (N,2), for semantic overlay")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ctrl = load_controller(args.run_dir)

    print(f"Evaluating {args.resolution}×{args.resolution} grid over z∈{args.z_range}…")
    z1, z2, actions = eval_grid(ctrl, args.resolution, tuple(args.z_range))
    print(f"Action range: [{actions.min():.2f}, {actions.max():.2f}]")

    traj_z = None
    if args.traj_npy and os.path.exists(args.traj_npy):
        traj_z = np.load(args.traj_npy)
        print(f"Loaded trajectory: {traj_z.shape} points")

    show_grid = not args.no_grid

    if args.mode in ("actions", "both"):
        out = os.path.join(args.out_dir, "latent2_action_heatmaps.png")
        plot_action_heatmaps(z1, z2, actions, args.quant_step, out, show_grid)

    if args.mode in ("semantic", "both"):
        out = os.path.join(args.out_dir, "latent2_semantic.png")
        plot_semantic_heatmap(z1, z2, actions, args.quant_step, out, traj_z, show_grid)

    if args.mode == "specoverlay":
        out = os.path.join(args.out_dir, "latent2_spec_overlay.pdf")
        plot_spec_overlay(z1, z2, actions, args.quant_step, out, show_grid)

    if args.mode == "cells":
        out = os.path.join(args.out_dir, "latent2_cells.pdf")
        plot_cell_grid(ctrl, args.quant_step, tuple(args.z_range), out)


if __name__ == "__main__":
    main()

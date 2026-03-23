from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sfm_pipeline.notebook_utils import (
    OBJECT_IDS,
    aggregate_metrics,
    load_doppelgangers_df,
    load_graph,
    load_metrics_df,
    load_point_cloud,
    make_contact_sheet,
    make_pair_strip,
    plot_image_graph,
    resolve_repo_root,
    sample_image_paths,
)


ROOT = resolve_repo_root(REPO_ROOT)
FIGS_DIR = ROOT / "report" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


POINT_CLOUD_VIEW_CONFIG = {
    "buda": {
        "view": (12, -120),
        "bounds_quantiles": (2, 98),
        "bounds_margin_scale": 1.24,
        "background": "#101215",
        "color_mode": "buda-dark",
        "point_size": 1.15,
    },
    "estatua": {
        "view": (12, -45),
        "bounds_quantiles": (1, 99),
        "bounds_margin_scale": 1.65,
        "background": "#ece3d4",
        "color_mode": "rgb-soft",
        "point_size": 0.95,
    },
    "leon": {
        "view": (18, 20),
        "bounds_quantiles": (2, 98),
        "bounds_margin_scale": 1.08,
        "background": "#ece3d4",
        "color_mode": "rgb-soft",
        "point_size": 1.05,
    },
}


def _viewer_coordinates(xyz: np.ndarray) -> np.ndarray:
    # Three.js uses Y as the up axis; matplotlib's 3D plots use Z as vertical.
    return np.column_stack([xyz[:, 0], xyz[:, 2], xyz[:, 1]])


def _viewer_colors(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "buda-dark":
        warm_white = np.array([0.93, 0.90, 0.86], dtype=float)
        return np.clip(0.22 * rgb + 0.78 * warm_white, 0.0, 1.0)
    if mode == "rgb-soft":
        gray = rgb.mean(axis=1, keepdims=True)
        return np.clip(0.72 * rgb + 0.28 * gray, 0.0, 1.0)
    return rgb


def _set_viewer_bounds(ax, xyz: np.ndarray, quantiles: tuple[float, float], margin_scale: float) -> tuple[np.ndarray, float]:
    low_q, high_q = quantiles
    mins = np.percentile(xyz, low_q, axis=0)
    maxs = np.percentile(xyz, high_q, axis=0)
    center = (mins + maxs) / 2.0
    radius = 0.5 * max(float(np.max(maxs - mins)), 1e-6) * max(margin_scale, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    return center, radius


def _draw_axes_helper(ax, center: np.ndarray, radius: float) -> None:
    origin = center + np.array([0.10 * radius, 0.02 * radius, 0.18 * radius], dtype=float)
    length = 0.30 * radius
    directions = [
        (np.array([1.0, 0.0, 0.0]), "#f59e0b"),
        (np.array([0.0, 1.0, 0.0]), "#84cc16"),
        (np.array([0.0, 0.0, 1.0]), "#38bdf8"),
    ]
    for direction, color in directions:
        end = origin + direction * length
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=color,
            linewidth=1.2,
            alpha=0.95,
            solid_capstyle="round",
        )


def save_object_samples(object_id: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sheet = make_contact_sheet(sample_image_paths(ROOT, object_id, max_images=6), cols=3)
    ax.imshow(sheet)
    ax.set_title(f"{object_id.capitalize()}: Representative Views", fontsize=15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"{object_id}_samples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_object_graph(object_id: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    nodes, edges = load_graph(ROOT, object_id, min_inliers=15)
    plot_image_graph(ax, nodes, edges, title=f"{object_id.capitalize()}: Image Connectivity", max_edges=180)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"{object_id}_graph.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_object_incremental(object_id: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    metrics_df = load_metrics_df(ROOT, object_id)
    aggregated = aggregate_metrics(metrics_df)
    full_row = metrics_df.loc[metrics_df["is_full"]].iloc[0]

    axes[0].plot(aggregated["num_images"], aggregated["registration_ratio"], marker="o", color="#2b6cb0")
    axes[0].fill_between(
        aggregated["num_images"],
        aggregated["registration_ratio"] - aggregated["registration_std"],
        aggregated["registration_ratio"] + aggregated["registration_std"],
        alpha=0.15,
        color="#2b6cb0",
    )
    axes[0].scatter([full_row["num_images"]], [full_row["registration_ratio"]], color="#111827", s=34)
    axes[0].set_title("Registration Ratio")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel(object_id.capitalize())

    axes[1].plot(aggregated["num_images"], aggregated["points3d"], marker="o", color="#2f855a")
    axes[1].fill_between(
        aggregated["num_images"],
        aggregated["points3d"] - aggregated["points3d_std"],
        aggregated["points3d"] + aggregated["points3d_std"],
        alpha=0.15,
        color="#2f855a",
    )
    axes[1].scatter([full_row["num_images"]], [full_row["points3d"]], color="#111827", s=34)
    axes[1].set_title("3D Points")

    axes[2].plot(aggregated["num_images"], aggregated["reproj_mean"], marker="o", color="#c05621")
    axes[2].fill_between(
        aggregated["num_images"],
        aggregated["reproj_mean"] - aggregated["reproj_std"],
        aggregated["reproj_mean"] + aggregated["reproj_std"],
        alpha=0.15,
        color="#c05621",
    )
    axes[2].scatter([full_row["num_images"]], [full_row["reproj_mean"]], color="#111827", s=34)
    axes[2].set_title("Mean Reprojection Error")

    for ax in axes:
        ax.set_xlabel("Number of Images")
    fig.suptitle(f"{object_id.capitalize()}: Incremental Reconstruction", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"{object_id}_incremental.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_object_pointcloud(object_id: str) -> None:
    config = POINT_CLOUD_VIEW_CONFIG.get(object_id, {})
    fig = plt.figure(figsize=(7.6, 4.35))
    fig.patch.set_facecolor(config.get("background", "#f5f1e8"))
    xyz, rgb, _ = load_point_cloud(ROOT, object_id, filtered=True, max_points=14000)
    xyz_view = _viewer_coordinates(xyz)
    colors = _viewer_colors(rgb, config.get("color_mode", "rgb"))

    ax = fig.add_subplot(1, 1, 1, projection="3d")
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("ortho")
    ax.scatter(
        xyz_view[:, 0],
        xyz_view[:, 1],
        xyz_view[:, 2],
        c=colors,
        s=float(config.get("point_size", 1.0)),
        alpha=0.92,
        linewidths=0,
    )
    center, radius = _set_viewer_bounds(
        ax,
        xyz_view,
        config.get("bounds_quantiles", (2, 98)),
        float(config.get("bounds_margin_scale", 1.0)),
    )
    _draw_axes_helper(ax, center, radius)
    view = config.get("view", (24, -58))
    ax.view_init(elev=float(view[0]), azim=float(view[1]))
    ax.set_facecolor(config.get("background", "#f5f1e8"))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(FIGS_DIR / f"{object_id}_pointcloud.png", dpi=220)
    plt.close(fig)


def save_object_doppelganger(object_id: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    doppel_df = load_doppelgangers_df(ROOT, object_id)
    row = doppel_df.iloc[0]
    strip = make_pair_strip(ROOT, object_id, row["image_a"], row["image_b"], size=(420, 320))
    ax.imshow(strip)
    ax.set_title(
        f"{object_id.capitalize()}: score={row['suspicion_score']:.0f}, "
        f"geom={row['geometric_score']:.3f}",
        fontsize=14,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"{object_id}_doppelganger.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for object_id in OBJECT_IDS:
        save_object_samples(object_id)
        save_object_graph(object_id)
        save_object_incremental(object_id)
        save_object_pointcloud(object_id)
        save_object_doppelganger(object_id)
    print(f"Saved figures to {FIGS_DIR}")


if __name__ == "__main__":
    main()

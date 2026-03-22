from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sfm_pipeline.notebook_utils import (
    OBJECT_IDS,
    aggregate_metrics,
    load_camera_centers,
    load_doppelgangers_df,
    load_graph,
    load_metrics_df,
    load_point_cloud,
    make_contact_sheet,
    make_pair_strip,
    plot_image_graph,
    plot_point_cloud,
    resolve_repo_root,
    sample_image_paths,
)


ROOT = resolve_repo_root(REPO_ROOT)
FIGS_DIR = ROOT / "report" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def save_dataset_samples() -> None:
    fig, axes = plt.subplots(1, len(OBJECT_IDS), figsize=(14, 4.5))
    for ax, object_id in zip(axes, OBJECT_IDS, strict=True):
        sheet = make_contact_sheet(sample_image_paths(ROOT, object_id, max_images=6), cols=3)
        ax.imshow(sheet)
        ax.set_title(f"{object_id.capitalize()}")
        ax.axis("off")
    fig.suptitle("Representative Views from the Three Datasets", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "dataset_samples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_view_graphs() -> None:
    fig, axes = plt.subplots(1, len(OBJECT_IDS), figsize=(14, 4.5))
    for ax, object_id in zip(axes, OBJECT_IDS, strict=True):
        nodes, edges = load_graph(ROOT, object_id, min_inliers=15)
        plot_image_graph(ax, nodes, edges, title=object_id.capitalize())
    fig.suptitle("Image Connectivity Graphs", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "view_graphs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_incremental_metrics() -> None:
    fig, axes = plt.subplots(len(OBJECT_IDS), 3, figsize=(12, 9), sharex="col")

    for row_index, object_id in enumerate(OBJECT_IDS):
        metrics_df = load_metrics_df(ROOT, object_id)
        aggregated = aggregate_metrics(metrics_df)
        full_row = metrics_df.loc[metrics_df["is_full"]].iloc[0]

        axes[row_index, 0].plot(
            aggregated["num_images"], aggregated["registration_ratio"], marker="o", color="#2b6cb0"
        )
        axes[row_index, 0].fill_between(
            aggregated["num_images"],
            aggregated["registration_ratio"] - aggregated["registration_std"],
            aggregated["registration_ratio"] + aggregated["registration_std"],
            alpha=0.15,
            color="#2b6cb0",
        )
        axes[row_index, 0].scatter([full_row["num_images"]], [full_row["registration_ratio"]], color="#111827", s=28)
        axes[row_index, 0].set_ylabel(object_id.capitalize())
        axes[row_index, 0].set_ylim(0, 1.05)

        axes[row_index, 1].plot(aggregated["num_images"], aggregated["points3d"], marker="o", color="#2f855a")
        axes[row_index, 1].fill_between(
            aggregated["num_images"],
            aggregated["points3d"] - aggregated["points3d_std"],
            aggregated["points3d"] + aggregated["points3d_std"],
            alpha=0.15,
            color="#2f855a",
        )
        axes[row_index, 1].scatter([full_row["num_images"]], [full_row["points3d"]], color="#111827", s=28)

        axes[row_index, 2].plot(aggregated["num_images"], aggregated["reproj_mean"], marker="o", color="#c05621")
        axes[row_index, 2].fill_between(
            aggregated["num_images"],
            aggregated["reproj_mean"] - aggregated["reproj_std"],
            aggregated["reproj_mean"] + aggregated["reproj_std"],
            alpha=0.15,
            color="#c05621",
        )
        axes[row_index, 2].scatter([full_row["num_images"]], [full_row["reproj_mean"]], color="#111827", s=28)

    axes[0, 0].set_title("Registration Ratio")
    axes[0, 1].set_title("3D Points")
    axes[0, 2].set_title("Mean Reprojection Error")
    for ax in axes[-1, :]:
        ax.set_xlabel("Number of Images")

    fig.tight_layout()
    fig.savefig(FIGS_DIR / "incremental_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_filtered_pointclouds() -> None:
    fig = plt.figure(figsize=(14, 4.8))
    for index, object_id in enumerate(OBJECT_IDS, start=1):
        xyz, rgb, _ = load_point_cloud(ROOT, object_id, filtered=True, max_points=12000)
        cameras = load_camera_centers(ROOT, object_id, filtered=True, max_cameras=180)
        ax = fig.add_subplot(1, len(OBJECT_IDS), index, projection="3d")
        plot_point_cloud(ax, xyz, rgb, cameras, title=object_id.capitalize())
    fig.suptitle("Filtered Sparse Models", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "filtered_pointclouds.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_doppelganger_examples() -> None:
    fig, axes = plt.subplots(1, len(OBJECT_IDS), figsize=(14, 4.5))
    for ax, object_id in zip(axes, OBJECT_IDS, strict=True):
        doppel_df = load_doppelgangers_df(ROOT, object_id)
        row = doppel_df.iloc[0]
        strip = make_pair_strip(ROOT, object_id, row["image_a"], row["image_b"])
        ax.imshow(strip)
        ax.set_title(
            f"{object_id.capitalize()}\n"
            f"score={row['suspicion_score']:.0f}, geom={row['geometric_score']:.3f}"
        )
        ax.axis("off")
    fig.suptitle("Representative Doppelganger Pairs", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "doppelganger_examples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    save_dataset_samples()
    save_view_graphs()
    save_incremental_metrics()
    save_filtered_pointclouds()
    save_doppelganger_examples()
    print(f"Saved figures to {FIGS_DIR}")


if __name__ == "__main__":
    main()

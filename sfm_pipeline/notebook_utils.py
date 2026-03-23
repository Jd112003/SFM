from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from sfm_pipeline.analysis import build_image_graph, collect_reconstruction_metrics, read_image_manifest


OBJECT_IDS = ("buda", "estatua", "leon")


def resolve_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "sfm_pipeline").exists():
            return candidate
        has_root_objects = any((candidate / object_id).exists() for object_id in OBJECT_IDS)
        has_data_objects = any((candidate / "data" / object_id).exists() for object_id in OBJECT_IDS)
        if (candidate / "outputs").exists() and (has_root_objects or has_data_objects):
            return candidate
    raise FileNotFoundError("Could not resolve repository root from the current working directory.")


def object_paths(root: Path, object_id: str) -> dict[str, Path]:
    data_object_dir = root / "data" / object_id
    if not data_object_dir.exists():
        legacy_object_dir = root / object_id
        data_object_dir = legacy_object_dir if legacy_object_dir.exists() else data_object_dir
    return {
        "images_dir": data_object_dir / "images",
        "outputs_dir": root / "outputs" / object_id,
        "metrics_path": root / "outputs" / object_id / "metrics" / "metrics.csv",
        "database_path": root / "outputs" / object_id / "database.db",
        "graph_edges_path": root / "outputs" / object_id / "graphs" / "image_graph_edges.csv",
        "doppel_path": root / "outputs" / object_id / "doppelgangers" / "doppelgangers.csv",
        "text_model_dir": root / "outputs" / object_id / "reconstruction" / "text_model",
        "filtered_model_dir": root / "outputs" / object_id / "reconstruction" / "filtered_text_model",
    }


def _as_dataframe(rows: list[dict[str, object]]):
    import pandas as pd

    return pd.DataFrame(rows)


def load_metrics_df(root: Path, object_id: str):
    import pandas as pd

    paths = object_paths(root, object_id)
    df = pd.read_csv(paths["metrics_path"])
    if df.empty:
        return df
    numeric_columns = [
        "num_images",
        "reproj_mean",
        "reproj_median",
        "registered_images",
        "registration_ratio",
        "points3d",
        "avg_track_length",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["success"] = df["success"].astype(str).str.lower().isin({"1", "true", "yes"})
    df["is_full"] = df["run_id"].eq("full")
    return df


def aggregate_metrics(metrics_df):
    if metrics_df.empty:
        return metrics_df
    subset = metrics_df.loc[~metrics_df["is_full"]].copy()
    if subset.empty:
        subset = metrics_df.copy()
    grouped = (
        subset.groupby("num_images", as_index=False)
        .agg(
            reproj_mean=("reproj_mean", "mean"),
            reproj_std=("reproj_mean", "std"),
            registration_ratio=("registration_ratio", "mean"),
            registration_std=("registration_ratio", "std"),
            points3d=("points3d", "mean"),
            points3d_std=("points3d", "std"),
        )
        .sort_values("num_images")
    )
    return grouped.fillna(0.0)


def build_dataset_summary(root: Path, object_ids: Iterable[str] = OBJECT_IDS):
    rows: list[dict[str, object]] = []
    for object_id in object_ids:
        paths = object_paths(root, object_id)
        images = read_image_manifest(paths["images_dir"])
        raw_metrics = collect_reconstruction_metrics(paths["text_model_dir"], len(images))
        filtered_metrics = collect_reconstruction_metrics(paths["filtered_model_dir"], len(images))
        suspects = load_doppelgangers_df(root, object_id)
        rows.append(
            {
                "object_id": object_id,
                "input_images": len(images),
                "registered_images": raw_metrics["registered_images"],
                "registration_ratio": raw_metrics["registration_ratio"],
                "raw_points3d": raw_metrics["points3d"],
                "filtered_points3d": filtered_metrics["points3d"],
                "raw_reproj_mean": raw_metrics["reproj_mean"],
                "filtered_reproj_mean": filtered_metrics["reproj_mean"],
                "avg_track_length": raw_metrics["avg_track_length"],
                "doppel_candidates": len(suspects),
            }
        )
    return _as_dataframe(rows).sort_values("object_id").reset_index(drop=True)


def sample_image_paths(root: Path, object_id: str, max_images: int = 6) -> list[Path]:
    images = read_image_manifest(object_paths(root, object_id)["images_dir"])
    if len(images) <= max_images:
        return images
    indices = np.linspace(0, len(images) - 1, num=max_images, dtype=int)
    return [images[index] for index in indices]


def make_contact_sheet(
    image_paths: list[Path],
    cols: int = 3,
    thumb_size: tuple[int, int] = (320, 240),
    background: tuple[int, int, int] = (245, 244, 239),
):
    from PIL import Image, ImageOps

    if not image_paths:
        raise ValueError("At least one image is required to create a contact sheet.")

    rows = math.ceil(len(image_paths) / cols)
    sheet = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1]), color=background)
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            tile = ImageOps.fit(image.convert("RGB"), thumb_size, method=Image.Resampling.LANCZOS)
        x = (index % cols) * thumb_size[0]
        y = (index // cols) * thumb_size[1]
        sheet.paste(tile, (x, y))
    return sheet


def load_graph(root: Path, object_id: str, min_inliers: int = 15):
    return build_image_graph(object_paths(root, object_id)["database_path"], min_inliers=min_inliers)


def plot_image_graph(ax, nodes, edges, title: str, max_edges: int = 220, label_nodes: bool = False) -> None:
    if not nodes:
        ax.set_title(title)
        ax.axis("off")
        return

    node_by_id = {node.image_id: node for node in nodes}
    palette = np.array(
        [
            [27, 158, 119],
            [217, 95, 2],
            [117, 112, 179],
            [102, 166, 30],
            [231, 41, 138],
            [230, 171, 2],
        ],
        dtype=float,
    ) / 255.0

    edge_subset = sorted(edges, key=lambda item: item.weight, reverse=True)[:max_edges]
    for edge in sorted(edge_subset, key=lambda item: item.inlier_ratio):
        node_a = node_by_id[edge.image_id1]
        node_b = node_by_id[edge.image_id2]
        tone = 0.25 + 0.45 * float(edge.inlier_ratio)
        ax.plot(
            [node_a.x, node_b.x],
            [node_a.y, node_b.y],
            linewidth=0.25 + 1.05 * min(edge.inlier_ratio, 1.0),
            color=(tone, tone * 0.95, tone * 0.82),
            alpha=0.10 + 0.16 * min(edge.inlier_ratio, 1.0),
            zorder=1,
        )

    colors = [palette[node.component % len(palette)] for node in nodes]
    sizes = [10 + min(node.degree, 18) * 1.7 for node in nodes]
    xs = [node.x for node in nodes]
    ys = [node.y for node in nodes]
    ax.scatter(xs, ys, s=sizes, c=colors, edgecolors="#24323a", linewidths=0.25, alpha=0.9, zorder=2)

    if label_nodes:
        for node in sorted(nodes, key=lambda item: item.degree, reverse=True)[:4]:
            ax.text(node.x, node.y, node.name.split(".")[0][-3:], fontsize=6, ha="center", va="center", zorder=3)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_facecolor("#fbf9f3")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _model_dir(root: Path, object_id: str, filtered: bool) -> Path:
    paths = object_paths(root, object_id)
    return paths["filtered_model_dir"] if filtered else paths["text_model_dir"]


def _points3d_path(root: Path, object_id: str, filtered: bool) -> Path:
    return _model_dir(root, object_id, filtered) / "points3D.txt"


def _images_txt_path(root: Path, object_id: str, filtered: bool) -> Path:
    return _model_dir(root, object_id, filtered) / "images.txt"


def load_point_cloud(
    root: Path,
    object_id: str,
    filtered: bool = False,
    max_points: int = 15_000,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyzs: list[tuple[float, float, float]] = []
    rgbs: list[tuple[int, int, int]] = []
    errors: list[float] = []

    with _points3d_path(root, object_id, filtered).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            xyzs.append((float(parts[1]), float(parts[2]), float(parts[3])))
            rgbs.append((int(parts[4]), int(parts[5]), int(parts[6])))
            errors.append(float(parts[7]))

    xyz = np.asarray(xyzs, dtype=float)
    rgb = np.asarray(rgbs, dtype=float) / 255.0
    reproj_error = np.asarray(errors, dtype=float)
    if len(xyz) > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(xyz), size=max_points, replace=False))
        xyz = xyz[keep]
        rgb = rgb[keep]
        reproj_error = reproj_error[keep]
    return xyz, rgb, reproj_error


def _quaternion_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def load_camera_centers(
    root: Path,
    object_id: str,
    filtered: bool = False,
    max_cameras: int | None = None,
) -> np.ndarray:
    centers: list[np.ndarray] = []
    with _images_txt_path(root, object_id, filtered).open("r", encoding="utf-8") as handle:
        raw_lines = handle.readlines()

    index = 0
    while index < len(raw_lines):
        line = raw_lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            index += 1
            continue
        qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], dtype=float)
        tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])], dtype=float)
        rotation = _quaternion_to_rotation_matrix(qvec)
        centers.append((-rotation.T @ tvec).astype(float))
        index += 2

    camera_centers = np.asarray(centers, dtype=float)
    if max_cameras is not None and len(camera_centers) > max_cameras:
        keep = np.linspace(0, len(camera_centers) - 1, num=max_cameras, dtype=int)
        camera_centers = camera_centers[keep]
    return camera_centers


def _set_axes_equal(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans.max(), 1e-6)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def _set_axes_equal_from_points(
    ax,
    xyz: np.ndarray,
    quantiles: tuple[float, float] | None = None,
    margin_scale: float = 1.0,
) -> None:
    if len(xyz) == 0:
        return
    if quantiles is None:
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
    else:
        low_q, high_q = quantiles
        mins = np.percentile(xyz, low_q, axis=0)
        maxs = np.percentile(xyz, high_q, axis=0)
    centers = (mins + maxs) / 2.0
    radius = 0.5 * max(float(np.max(maxs - mins)), 1e-6) * max(margin_scale, 1e-6)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_point_cloud(
    ax,
    xyz: np.ndarray,
    rgb: np.ndarray,
    cameras: np.ndarray | None,
    title: str,
    view: tuple[float, float] | None = None,
    bounds_quantiles: tuple[float, float] | None = None,
    bounds_margin_scale: float = 1.0,
    show_axis_labels: bool = True,
    show_ticks: bool = True,
) -> None:
    if len(xyz) > 0:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1.8, alpha=0.85, linewidths=0)
    if cameras is not None and len(cameras) > 0:
        ax.scatter(
            cameras[:, 0],
            cameras[:, 1],
            cameras[:, 2],
            c="#0d2f4f",
            s=5,
            alpha=0.35,
            depthshade=False,
            label="Cameras",
        )
    ax.set_title(title)
    if show_axis_labels:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    ax.set_facecolor("#fbf9f3")
    elev, azim = view if view is not None else (24, -58)
    ax.view_init(elev=elev, azim=azim)
    _set_axes_equal_from_points(ax, xyz, quantiles=bounds_quantiles, margin_scale=bounds_margin_scale)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False


def load_doppelgangers_df(root: Path, object_id: str):
    import pandas as pd

    path = object_paths(root, object_id)["doppel_path"]
    if not path.exists():
        return pd.DataFrame(
            columns=["image_a", "image_b", "visual_score", "geometric_score", "suspicion_score", "reason"]
        )
    df = pd.read_csv(path)
    for column in ("visual_score", "geometric_score", "suspicion_score"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def make_pair_strip(
    root: Path,
    object_id: str,
    image_a: str,
    image_b: str,
    size: tuple[int, int] = (320, 320),
):
    from PIL import Image, ImageOps

    images_dir = object_paths(root, object_id)["images_dir"]
    output = Image.new("RGB", (size[0] * 2, size[1]), color=(245, 244, 239))
    for column, filename in enumerate((image_a, image_b)):
        with Image.open(images_dir / filename) as image:
            image = ImageOps.exif_transpose(image)
            tile = ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)
        output.paste(tile, (column * size[0], 0))
    return output

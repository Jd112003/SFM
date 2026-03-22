from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sqlite3
from pathlib import Path

from sfm_pipeline.analysis import (
    append_metrics_row,
    build_image_graph,
    collect_reconstruction_metrics,
    create_doppelganger_gallery,
    detect_doppelgangers,
    export_graph_dot,
    export_graph_html,
    filter_sparse_text_model,
    load_metrics,
    read_image_manifest,
    write_doppelgangers,
    write_edges_csv,
)
from sfm_pipeline.colmap import ColmapNotAvailableError, export_text_model, feature_extract, map_sparse, match_features
from sfm_pipeline.config import load_config, resolve_paths
from sfm_pipeline.plots import write_metrics_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SfM object pipeline")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")
    common.add_argument("--object", dest="object_id", required=True, help="Object identifier under the data directory")
    common.add_argument("--seed", type=int, default=7, help="Random seed for ablations")
    common.add_argument("--overwrite", action="store_true", help="Overwrite generated files when needed")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in [
        "prepare",
        "extract-match",
        "reconstruct",
        "clean-model",
        "analyze-graph",
        "detect-doppelgangers",
        "run-ablation",
        "report",
    ]:
        subparsers.add_parser(name, parents=[common])
    return parser.parse_args()


def object_paths(config: dict, object_id: str, root_dir: Path) -> dict[str, Path]:
    resolved = resolve_paths(config, root_dir=root_dir)
    data_dir = resolved["data_dir"] / object_id
    images_dir = data_dir / "images"
    outputs_dir = resolved["outputs_dir"] / object_id
    return {
        "root": resolved["root"],
        "data_dir": data_dir,
        "images_dir": images_dir,
        "metadata_path": data_dir / "metadata.yaml",
        "outputs_dir": outputs_dir,
        "database_path": outputs_dir / "database.db",
        "sparse_dir": outputs_dir / "reconstruction" / "sparse",
        "model_text_dir": outputs_dir / "reconstruction" / "text_model",
        "filtered_model_text_dir": outputs_dir / "reconstruction" / "filtered_text_model",
        "graphs_dir": outputs_dir / "graphs",
        "metrics_dir": outputs_dir / "metrics",
        "doppel_dir": outputs_dir / "doppelgangers",
        "doppel_database_path": outputs_dir / "doppelgangers" / "database.db",
        "ablation_dir": outputs_dir / "ablation",
    }


def ensure_dataset(paths: dict[str, Path]) -> list[Path]:
    if not paths["images_dir"].exists():
        raise FileNotFoundError(f"Image directory not found: {paths['images_dir']}")
    images = read_image_manifest(paths["images_dir"])
    if not images:
        raise FileNotFoundError(f"No supported images found in {paths['images_dir']}")
    return images


def ensure_database(paths: dict[str, Path]) -> Path:
    if not paths["database_path"].exists():
        raise FileNotFoundError(
            f"COLMAP database not found: {paths['database_path']}. Run extract-match first."
        )
    return paths["database_path"]


def cmd_prepare(config: dict, paths: dict[str, Path], overwrite: bool) -> None:
    images = ensure_dataset(paths)
    paths["outputs_dir"].mkdir(parents=True, exist_ok=True)
    manifest_path = paths["outputs_dir"] / "image_manifest.csv"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {manifest_path}. Use --overwrite to replace it.")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "filename", "relative_path"])
        writer.writeheader()
        for index, image_path in enumerate(images, start=1):
            writer.writerow(
                {
                    "index": index,
                    "filename": image_path.name,
                    "relative_path": image_path.relative_to(paths["data_dir"]).as_posix(),
                }
            )
    if not paths["metadata_path"].exists():
        paths["metadata_path"].write_text(
            "name: \nlocation: \ncamera: \nnotes: \n",
            encoding="utf-8",
        )


def cmd_extract_match(config: dict, paths: dict[str, Path]) -> None:
    ensure_dataset(paths)
    paths["outputs_dir"].mkdir(parents=True, exist_ok=True)
    feature_extract(paths["database_path"], paths["images_dir"], config["colmap"])
    match_features(paths["database_path"], config["colmap"])


def cmd_reconstruct(config: dict, paths: dict[str, Path], object_id: str) -> None:
    images = ensure_dataset(paths)
    ensure_database(paths)
    model_dir = map_sparse(paths["database_path"], paths["images_dir"], paths["sparse_dir"], config["colmap"])
    export_text_model(model_dir, paths["model_text_dir"])
    metrics = collect_reconstruction_metrics(paths["model_text_dir"], len(images))
    append_metrics_row(
        paths["metrics_dir"] / "metrics.csv",
        {
            "object_id": object_id,
            "run_id": "full",
            "num_images": len(images),
            **metrics,
        },
    )


def cmd_clean_model(config: dict, paths: dict[str, Path]) -> None:
    if not paths["model_text_dir"].exists():
        raise FileNotFoundError(
            f"Text model not found: {paths['model_text_dir']}. Run reconstruct first."
        )
    filter_sparse_text_model(
        paths["model_text_dir"],
        paths["filtered_model_text_dir"],
        config.get("filtering", {}),
    )


def cmd_analyze_graph(config: dict, paths: dict[str, Path]) -> None:
    database_path = ensure_database(paths)
    nodes, edges = build_image_graph(database_path, min_inliers=config["graph"]["min_inliers"])
    export_graph_dot(nodes, edges, paths["graphs_dir"] / "image_graph.dot")
    export_graph_html(nodes, edges, paths["graphs_dir"] / "image_graph.html")
    write_edges_csv(edges, paths["graphs_dir"] / "image_graph_edges.csv")


def cmd_detect_doppelgangers(config: dict, paths: dict[str, Path]) -> None:
    database_path = ensure_database(paths)
    doppel_config = dict(config["doppelgangers"])
    matcher = str(doppel_config.get("matcher", "exhaustive")).lower()
    graph_database_path = database_path

    if matcher == "exhaustive":
        paths["doppel_dir"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(database_path, paths["doppel_database_path"])
        with sqlite3.connect(paths["doppel_database_path"]) as conn:
            conn.execute("DELETE FROM matches")
            conn.execute("DELETE FROM two_view_geometries")
            conn.commit()
        match_features(paths["doppel_database_path"], {**config["colmap"], "matcher": "exhaustive"})
        graph_database_path = paths["doppel_database_path"]

    _, edges = build_image_graph(graph_database_path, min_inliers=0)
    suspects = detect_doppelgangers(
        edges,
        min_visual_matches=doppel_config["min_visual_matches"],
        max_inlier_ratio=doppel_config["max_inlier_ratio"],
        top_k=doppel_config["top_k"],
    )
    write_doppelgangers(
        suspects,
        paths["doppel_dir"] / "doppelgangers.csv",
        paths["doppel_dir"] / "doppelgangers.json",
    )
    create_doppelganger_gallery(suspects, paths["images_dir"], paths["doppel_dir"] / "gallery.html")


def _prepare_subset_run(source_images: list[Path], destination_dir: Path, subset_size: int, seed: int) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    subset = random.Random(seed).sample(source_images, subset_size)
    copied: list[Path] = []
    for source in subset:
        destination = destination_dir / source.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        try:
            relative_target = os.path.relpath(source.resolve(), destination_dir.resolve())
            destination.symlink_to(relative_target)
        except OSError:
            shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _metrics_has_run_id(metrics_path: Path, run_id: str) -> bool:
    if not metrics_path.exists():
        return False
    rows = load_metrics(metrics_path)
    return any(row.get("run_id") == run_id for row in rows)


def _record_full_metrics_if_needed(paths: dict[str, Path], object_id: str, num_images: int) -> None:
    if not paths["model_text_dir"].exists():
        return
    metrics_path = paths["metrics_dir"] / "metrics.csv"
    if _metrics_has_run_id(metrics_path, "full"):
        return
    metrics = collect_reconstruction_metrics(paths["model_text_dir"], num_images)
    append_metrics_row(
        metrics_path,
        {
            "object_id": object_id,
            "run_id": "full",
            "num_images": num_images,
            **metrics,
        },
    )


def cmd_run_ablation(config: dict, paths: dict[str, Path], object_id: str, seed: int) -> None:
    all_images = ensure_dataset(paths)
    subset_sizes = sorted(set(config["ablation"]["subset_sizes"] + [len(all_images)]))
    runs_per_size = int(config["ablation"]["runs_per_size"])
    metrics_path = paths["metrics_dir"] / "metrics.csv"
    full_model_exists = paths["model_text_dir"].exists()

    if full_model_exists:
        _record_full_metrics_if_needed(paths, object_id, len(all_images))

    for subset_size in subset_sizes:
        if subset_size > len(all_images) or subset_size < 2:
            continue
        if subset_size == len(all_images) and full_model_exists:
            continue
        num_runs = 1 if subset_size == len(all_images) else runs_per_size
        for run_index in range(num_runs):
            run_id = f"n{subset_size}_r{run_index + 1}"
            run_dir = paths["ablation_dir"] / run_id
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_images_dir = run_dir / "images"
            run_database_path = run_dir / "database.db"
            run_sparse_dir = run_dir / "sparse"
            run_model_dir = run_dir / "text_model"

            _prepare_subset_run(all_images, run_images_dir, subset_size, seed + subset_size * 100 + run_index)
            feature_extract(run_database_path, run_images_dir, config["colmap"])
            match_features(run_database_path, config["colmap"])
            model_dir = map_sparse(run_database_path, run_images_dir, run_sparse_dir, config["colmap"])
            export_text_model(model_dir, run_model_dir)
            metrics = collect_reconstruction_metrics(run_model_dir, subset_size)
            append_metrics_row(
                metrics_path,
                {
                    "object_id": object_id,
                    "run_id": run_id,
                    "num_images": subset_size,
                    **metrics,
                },
            )


def cmd_report(config: dict, paths: dict[str, Path]) -> None:
    metrics_path = paths["metrics_dir"] / "metrics.csv"
    rows = load_metrics(metrics_path)
    if not rows:
        raise FileNotFoundError(f"No metrics found in {metrics_path}")
    write_metrics_plots(
        rows,
        paths["metrics_dir"] / "plots",
        include_success_plot=bool(config["report"]["include_success_plot"]),
    )


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
        root_dir = config_path.resolve().parent
    elif args.config != "config.yaml":
        raise SystemExit(f"Config file not found: {config_path}")
    else:
        config = load_config(None)
        root_dir = Path.cwd()
    paths = object_paths(config, args.object_id, root_dir)

    try:
        if args.command == "prepare":
            cmd_prepare(config, paths, args.overwrite)
        elif args.command == "extract-match":
            cmd_extract_match(config, paths)
        elif args.command == "reconstruct":
            cmd_reconstruct(config, paths, args.object_id)
        elif args.command == "clean-model":
            cmd_clean_model(config, paths)
        elif args.command == "analyze-graph":
            cmd_analyze_graph(config, paths)
        elif args.command == "detect-doppelgangers":
            cmd_detect_doppelgangers(config, paths)
        elif args.command == "run-ablation":
            cmd_run_ablation(config, paths, args.object_id, args.seed)
        elif args.command == "report":
            cmd_report(config, paths)
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except ColmapNotAvailableError as exc:
        raise SystemExit(str(exc)) from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

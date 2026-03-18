from __future__ import annotations

from pathlib import Path

from sfm_pipeline.cli import cmd_prepare, object_paths
from sfm_pipeline.config import load_config
from sfm_pipeline.plots import write_metrics_plots


def test_prepare_creates_manifest_and_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "obj" / "images"
    data_dir.mkdir(parents=True)
    for name in ("a.jpg", "b.png"):
        (data_dir / name).write_bytes(b"fake")

    config = load_config(None)
    paths = object_paths(config, "obj", tmp_path)
    cmd_prepare(config, paths, overwrite=False)

    assert (paths["outputs_dir"] / "image_manifest.csv").exists()
    assert paths["metadata_path"].exists()


def test_write_metrics_plots(tmp_path: Path) -> None:
    rows = [
        {
            "object_id": "obj",
            "run_id": "n6_r1",
            "num_images": "6",
            "success": "true",
            "reproj_mean": "1.2",
            "reproj_median": "1.0",
            "registered_images": "6",
            "registration_ratio": "1.0",
            "points3d": "120",
            "avg_track_length": "3.4",
        },
        {
            "object_id": "obj",
            "run_id": "n10_r1",
            "num_images": "10",
            "success": "true",
            "reproj_mean": "0.8",
            "reproj_median": "0.7",
            "registered_images": "10",
            "registration_ratio": "1.0",
            "points3d": "240",
            "avg_track_length": "4.1",
        },
    ]
    write_metrics_plots(rows, tmp_path / "plots", include_success_plot=True)
    assert (tmp_path / "plots" / "error_vs_num_images_mean.svg").exists()
    assert (tmp_path / "plots" / "success_rate_vs_num_images.svg").exists()

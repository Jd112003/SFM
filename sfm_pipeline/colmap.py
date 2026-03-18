from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence


class ColmapNotAvailableError(RuntimeError):
    pass


def ensure_colmap() -> str:
    colmap_bin = shutil.which("colmap")
    if colmap_bin is None:
        raise ColmapNotAvailableError(
            "COLMAP is not installed or not available in PATH. Install it before running reconstruction stages."
        )
    return colmap_bin


def run_colmap(command: Sequence[str], cwd: Path | None = None) -> None:
    ensure_colmap()
    subprocess.run(command, cwd=cwd, check=True)


def _gpu_flag(config: dict, key_prefix: str) -> list[str]:
    use_gpu = "1" if bool(config.get("use_gpu", False)) else "0"
    gpu_index = str(config.get("gpu_index", 0))
    return [f"--{key_prefix}.use_gpu", use_gpu, f"--{key_prefix}.gpu_index", gpu_index]


def feature_extract(database_path: Path, image_path: Path, config: dict) -> None:
    run_colmap(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_path),
            "--ImageReader.camera_model",
            str(config["camera_model"]),
            "--FeatureExtraction.max_image_size",
            str(config.get("max_image_size", -1)),
            "--SiftExtraction.max_num_features",
            str(config["sift_max_num_features"]),
            *_gpu_flag(config, "FeatureExtraction"),
        ]
    )


def match_features(database_path: Path, config: dict) -> None:
    matcher = config["matcher"].lower()
    if matcher == "sequential":
        run_colmap(
            [
                "colmap",
                "sequential_matcher",
                "--database_path",
                str(database_path),
                "--SequentialMatching.overlap",
                str(config["sequential_overlap"]),
                *_gpu_flag(config, "FeatureMatching"),
            ]
        )
        return

    run_colmap(
        [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            *_gpu_flag(config, "FeatureMatching"),
        ]
    )


def map_sparse(database_path: Path, image_path: Path, sparse_dir: Path, config: dict) -> Path:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    run_colmap(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_path),
            "--output_path",
            str(sparse_dir),
            "--Mapper.min_num_matches",
            str(config["mapper_min_num_matches"]),
        ]
    )
    models = sorted(p for p in sparse_dir.iterdir() if p.is_dir())
    if not models:
        raise RuntimeError(f"No sparse model generated in {sparse_dir}")
    return max(models, key=lambda path: path.stat().st_mtime)


def export_text_model(input_model_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_colmap(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(input_model_dir),
            "--output_path",
            str(output_dir),
            "--output_type",
            "TXT",
        ]
    )
    return output_dir

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "data_dir": "data",
        "outputs_dir": "outputs",
    },
    "colmap": {
        "camera_model": "SIMPLE_RADIAL",
        "matcher": "exhaustive",
        "use_gpu": False,
        "gpu_index": 0,
        "max_image_size": -1,
        "sift_max_num_features": 8192,
        "sequential_overlap": 10,
        "mapper_min_num_matches": 15,
    },
    "ablation": {
        "subset_sizes": [6, 10, 15, 20, 30],
        "runs_per_size": 3,
    },
    "graph": {
        "min_inliers": 15,
        "min_inlier_ratio": 0.15,
    },
    "doppelgangers": {
        "min_visual_matches": 80,
        "max_inlier_ratio": 0.25,
        "top_k": 15,
    },
    "report": {
        "include_success_plot": True,
    },
}


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return deepcopy(DEFAULT_CONFIG)
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected dict-like YAML config in {path}")
    return _merge_dicts(DEFAULT_CONFIG, loaded)


def resolve_paths(config: dict[str, Any], root_dir: str | Path | None = None) -> dict[str, Path]:
    base = Path(root_dir) if root_dir else Path.cwd()
    paths_cfg = config["paths"]
    data_dir = (base / paths_cfg["data_dir"]).resolve()
    outputs_dir = (base / paths_cfg["outputs_dir"]).resolve()
    return {"root": base.resolve(), "data_dir": data_dir, "outputs_dir": outputs_dir}

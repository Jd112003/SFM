from __future__ import annotations

import sqlite3
from pathlib import Path

from sfm_pipeline.analysis import (
    append_metrics_row,
    build_image_graph,
    collect_reconstruction_metrics,
    detect_doppelgangers,
    load_metrics,
)


def _create_test_database(database_path: Path) -> None:
    with sqlite3.connect(database_path) as conn:
        conn.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
        conn.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
        conn.executemany(
            "INSERT INTO images (image_id, name) VALUES (?, ?)",
            [(1, "img1.jpg"), (2, "img2.jpg"), (3, "img3.jpg")],
        )
        pair_12 = 1 * 2_147_483_647 + 2
        pair_23 = 2 * 2_147_483_647 + 3
        conn.execute("INSERT INTO matches (pair_id, rows) VALUES (?, ?)", (pair_12, 120))
        conn.execute("INSERT INTO matches (pair_id, rows) VALUES (?, ?)", (pair_23, 90))
        conn.execute("INSERT INTO two_view_geometries (pair_id, rows) VALUES (?, ?)", (pair_12, 18))
        conn.execute("INSERT INTO two_view_geometries (pair_id, rows) VALUES (?, ?)", (pair_23, 70))


def test_graph_and_doppelganger_detection(tmp_path: Path) -> None:
    database_path = tmp_path / "database.db"
    _create_test_database(database_path)

    nodes, edges = build_image_graph(database_path, min_inliers=10)
    assert len(nodes) == 3
    assert len(edges) == 2
    assert {node.degree for node in nodes} == {1, 2}

    suspects = detect_doppelgangers(edges, min_visual_matches=100, max_inlier_ratio=0.2, top_k=10)
    assert len(suspects) == 1
    assert suspects[0].image_a == "img1.jpg"
    assert suspects[0].image_b == "img2.jpg"


def test_collect_metrics_and_append(tmp_path: Path) -> None:
    model_dir = tmp_path / "text_model"
    model_dir.mkdir()
    (model_dir / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 a.jpg\n"
        "0 1 2\n"
        "2 1 0 0 0 0 0 0 1 b.jpg\n"
        "0 3 4\n",
        encoding="utf-8",
    )
    (model_dir / "points3D.txt").write_text(
        "# Points\n"
        "1 0 0 0 255 255 255 0.5 1 10 2 11\n"
        "2 0 0 0 255 255 255 1.5 1 12 2 13 3 14\n",
        encoding="utf-8",
    )

    metrics = collect_reconstruction_metrics(model_dir, num_input_images=4)
    assert metrics["success"] is True
    assert metrics["registered_images"] == 2
    assert metrics["points3d"] == 2

    metrics_csv = tmp_path / "metrics.csv"
    append_metrics_row(
        metrics_csv,
        {
            "object_id": "obj",
            "run_id": "full",
            "num_images": 4,
            **metrics,
        },
    )
    rows = load_metrics(metrics_csv)
    assert len(rows) == 1
    assert rows[0]["object_id"] == "obj"

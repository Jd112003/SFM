from __future__ import annotations

import csv
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np


COLMAP_MAX_IMAGE_ID = 2_147_483_647


@dataclass(slots=True)
class ImageNode:
    image_id: int
    name: str
    degree: int = 0
    component: int = -1
    x: float = 0.0
    y: float = 0.0


@dataclass(slots=True)
class ImageEdge:
    image_id1: int
    image_id2: int
    image_name1: str
    image_name2: str
    raw_matches: int
    inlier_matches: int
    inlier_ratio: float
    weight: float


@dataclass(slots=True)
class DoppelgangerPair:
    image_a: str
    image_b: str
    visual_score: float
    geometric_score: float
    suspicion_score: float
    reason: str


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    image_id2 = pair_id % COLMAP_MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // COLMAP_MAX_IMAGE_ID
    return image_id1, image_id2


def read_image_manifest(images_dir: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported
    )


def build_image_graph(database_path: Path, min_inliers: int = 0) -> tuple[list[ImageNode], list[ImageEdge]]:
    with sqlite3.connect(database_path) as conn:
        images = {
            int(image_id): name
            for image_id, name in conn.execute("SELECT image_id, name FROM images ORDER BY image_id")
        }

        raw_match_rows = {
            int(pair_id): int(rows or 0)
            for pair_id, rows in conn.execute("SELECT pair_id, rows FROM matches")
        }

        edges: list[ImageEdge] = []
        for pair_id, rows in conn.execute("SELECT pair_id, rows FROM two_view_geometries"):
            image_id1, image_id2 = pair_id_to_image_ids(int(pair_id))
            inliers = int(rows or 0)
            if inliers < min_inliers:
                continue
            raw = raw_match_rows.get(int(pair_id), inliers)
            ratio = (inliers / raw) if raw else 0.0
            weight = float(inliers * max(ratio, 1e-6))
            edges.append(
                ImageEdge(
                    image_id1=image_id1,
                    image_id2=image_id2,
                    image_name1=images.get(image_id1, f"image_{image_id1}"),
                    image_name2=images.get(image_id2, f"image_{image_id2}"),
                    raw_matches=raw,
                    inlier_matches=inliers,
                    inlier_ratio=ratio,
                    weight=weight,
                )
            )

    node_map = {image_id: ImageNode(image_id=image_id, name=name) for image_id, name in images.items()}
    for edge in edges:
        node_map[edge.image_id1].degree += 1
        node_map[edge.image_id2].degree += 1

    _assign_components(node_map, edges)
    _assign_layout(node_map, edges)
    return sorted(node_map.values(), key=lambda node: node.image_id), edges


def _assign_components(node_map: dict[int, ImageNode], edges: Iterable[ImageEdge]) -> None:
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in node_map}
    for edge in edges:
        adjacency[edge.image_id1].add(edge.image_id2)
        adjacency[edge.image_id2].add(edge.image_id1)

    component_id = 0
    visited: set[int] = set()
    for node_id in adjacency:
        if node_id in visited:
            continue
        stack = [node_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            node_map[current].component = component_id
            stack.extend(adjacency[current] - visited)
        component_id += 1


def _assign_layout(node_map: dict[int, ImageNode], edges: list[ImageEdge]) -> None:
    if not node_map:
        return
    image_ids = list(node_map)
    index = {image_id: i for i, image_id in enumerate(image_ids)}
    n = len(image_ids)
    positions = np.zeros((n, 2), dtype=float)

    for i in range(n):
        angle = (2.0 * math.pi * i) / max(n, 1)
        positions[i] = np.array([math.cos(angle), math.sin(angle)], dtype=float)

    if edges:
        adjacency = np.zeros((n, n), dtype=float)
        for edge in edges:
            i = index[edge.image_id1]
            j = index[edge.image_id2]
            adjacency[i, j] = edge.weight
            adjacency[j, i] = edge.weight

        iterations = min(100, max(30, n * 5))
        repulsion = 0.12
        attraction = 0.005
        for _ in range(iterations):
            delta = np.zeros_like(positions)
            for i in range(n):
                diff = positions[i] - positions
                dist = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
                delta[i] += np.sum((diff / dist) * (repulsion / dist), axis=0)
                neighbors = np.where(adjacency[i] > 0)[0]
                for j in neighbors:
                    delta[i] -= attraction * adjacency[i, j] * (positions[i] - positions[j])
            positions += delta
            max_norm = np.max(np.linalg.norm(positions, axis=1)) or 1.0
            positions /= max_norm

    for image_id, pos in zip(image_ids, positions, strict=True):
        node_map[image_id].x = float(pos[0])
        node_map[image_id].y = float(pos[1])


def detect_doppelgangers(
    edges: Iterable[ImageEdge],
    min_visual_matches: int,
    max_inlier_ratio: float,
    top_k: int,
) -> list[DoppelgangerPair]:
    suspects: list[DoppelgangerPair] = []
    for edge in edges:
        if edge.raw_matches < min_visual_matches:
            continue
        if edge.inlier_ratio > max_inlier_ratio:
            continue
        visual_score = float(edge.raw_matches)
        geometric_score = float(edge.inlier_ratio)
        suspicion = visual_score * (1.0 - geometric_score)
        reason = (
            f"Many raw matches ({edge.raw_matches}) but weak geometric agreement "
            f"(inlier ratio {edge.inlier_ratio:.2f})"
        )
        suspects.append(
            DoppelgangerPair(
                image_a=edge.image_name1,
                image_b=edge.image_name2,
                visual_score=visual_score,
                geometric_score=geometric_score,
                suspicion_score=suspicion,
                reason=reason,
            )
        )
    suspects.sort(key=lambda item: item.suspicion_score, reverse=True)
    return suspects[:top_k]


def write_edges_csv(edges: Iterable[ImageEdge], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_a",
                "image_b",
                "raw_matches",
                "inlier_matches",
                "inlier_ratio",
                "weight",
            ],
        )
        writer.writeheader()
        for edge in edges:
            writer.writerow(
                {
                    "image_a": edge.image_name1,
                    "image_b": edge.image_name2,
                    "raw_matches": edge.raw_matches,
                    "inlier_matches": edge.inlier_matches,
                    "inlier_ratio": f"{edge.inlier_ratio:.6f}",
                    "weight": f"{edge.weight:.6f}",
                }
            )


def export_graph_dot(nodes: Iterable[ImageNode], edges: Iterable[ImageEdge], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#66a61e", "#e7298a", "#e6ab02"]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("graph image_relationships {\n")
        handle.write('  graph [layout=neato, overlap=false, splines=true];\n')
        handle.write('  node [shape=circle, style=filled, fontname="Helvetica"];\n')
        for node in nodes:
            color = palette[node.component % len(palette)]
            handle.write(
                f'  "{node.name}" [fillcolor="{color}", pos="{node.x * 10:.3f},{node.y * 10:.3f}!", '
                f'label="{node.name}\\n(d={node.degree})"];\n'
            )
        for edge in edges:
            penwidth = max(1.0, edge.inlier_ratio * 6.0)
            handle.write(
                f'  "{edge.image_name1}" -- "{edge.image_name2}" '
                f'[label="{edge.inlier_matches}", penwidth={penwidth:.2f}];\n'
            )
        handle.write("}\n")


def export_graph_html(nodes: Iterable[ImageNode], edges: Iterable[ImageEdge], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    node_payload = [asdict(node) for node in nodes]
    edge_payload = [asdict(edge) for edge in edges]
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Image Graph</title>
  <style>
    body {{
      margin: 0;
      background: #f6f4ef;
      color: #1c1c1c;
      font-family: Georgia, "Times New Roman", serif;
    }}
    #wrap {{
      display: grid;
      grid-template-columns: 1fr 320px;
      min-height: 100vh;
    }}
    #canvas {{
      width: 100%;
      height: 100vh;
      background: radial-gradient(circle at top, #fff9ed, #efe8db);
    }}
    aside {{
      padding: 24px;
      border-left: 1px solid #d9cfbc;
      background: rgba(255, 252, 246, 0.92);
    }}
    h1 {{
      margin-top: 0;
      font-size: 1.5rem;
    }}
    .card {{
      padding: 12px;
      margin-bottom: 12px;
      border: 1px solid #d9cfbc;
      background: #fffdf8;
    }}
    .small {{
      color: #555;
      font-size: 0.9rem;
    }}
  </style>
</head>
<body>
  <div id="wrap">
    <svg id="canvas" viewBox="-1.2 -1.2 2.4 2.4"></svg>
    <aside>
      <h1>Image Relationship Graph</h1>
      <div class="card small">
        Hover a node or edge to inspect connectivity, inliers and suspicious pairings.
      </div>
      <div id="info" class="card small">Select an element to inspect.</div>
    </aside>
  </div>
  <script>
    const nodes = {json.dumps(node_payload, ensure_ascii=True)};
    const edges = {json.dumps(edge_payload, ensure_ascii=True)};
    const svg = document.getElementById("canvas");
    const info = document.getElementById("info");
    const palette = ["#1b9e77", "#d95f02", "#7570b3", "#66a61e", "#e7298a", "#e6ab02"];

    function line(x1, y1, x2, y2, width, color) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
      el.setAttribute("x1", x1);
      el.setAttribute("y1", y1);
      el.setAttribute("x2", x2);
      el.setAttribute("y2", y2);
      el.setAttribute("stroke", color);
      el.setAttribute("stroke-width", width);
      el.setAttribute("opacity", "0.55");
      return el;
    }}

    function circle(cx, cy, r, fill) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      el.setAttribute("cx", cx);
      el.setAttribute("cy", cy);
      el.setAttribute("r", r);
      el.setAttribute("fill", fill);
      el.setAttribute("stroke", "#111");
      el.setAttribute("stroke-width", "0.01");
      return el;
    }}

    function text(x, y, value) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
      el.setAttribute("x", x);
      el.setAttribute("y", y);
      el.setAttribute("font-size", "0.06");
      el.setAttribute("text-anchor", "middle");
      el.setAttribute("font-family", "Georgia, serif");
      el.textContent = value;
      return el;
    }}

    const nodeById = new Map(nodes.map((node) => [node.image_id, node]));
    edges.forEach((edge) => {{
      const a = nodeById.get(edge.image_id1);
      const b = nodeById.get(edge.image_id2);
      const width = Math.max(0.01, edge.inlier_ratio * 0.08);
      const el = line(a.x, a.y, b.x, b.y, width, "#7a6d59");
      el.addEventListener("mouseenter", () => {{
        info.innerHTML = `<strong>${{edge.image_name1}}</strong> ↔ <strong>${{edge.image_name2}}</strong><br>` +
          `Raw matches: ${{edge.raw_matches}}<br>` +
          `Inliers: ${{edge.inlier_matches}}<br>` +
          `Inlier ratio: ${{edge.inlier_ratio.toFixed(3)}}`;
      }});
      svg.appendChild(el);
    }});

    nodes.forEach((node) => {{
      const fill = palette[node.component % palette.length];
      const el = circle(node.x, node.y, 0.045 + Math.min(node.degree, 8) * 0.005, fill);
      el.addEventListener("mouseenter", () => {{
        info.innerHTML = `<strong>${{node.name}}</strong><br>Image ID: ${{node.image_id}}<br>` +
          `Degree: ${{node.degree}}<br>Component: ${{node.component}}`;
      }});
      svg.appendChild(el);
      svg.appendChild(text(node.x, node.y - 0.07, node.name));
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def write_doppelgangers(
    suspects: Iterable[DoppelgangerPair],
    csv_path: Path,
    json_path: Path,
) -> None:
    suspects = list(suspects)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_a",
                "image_b",
                "visual_score",
                "geometric_score",
                "suspicion_score",
                "reason",
            ],
        )
        writer.writeheader()
        for item in suspects:
            writer.writerow(asdict(item))
    json_path.write_text(
        json.dumps([asdict(item) for item in suspects], indent=2),
        encoding="utf-8",
    )


def create_doppelganger_gallery(
    suspects: Iterable[DoppelgangerPair],
    images_dir: Path,
    output_path: Path,
) -> None:
    cards = []
    for item in suspects:
        image_a = (images_dir / item.image_a).as_posix()
        image_b = (images_dir / item.image_b).as_posix()
        cards.append(
            f"""
            <article class="pair">
              <div class="meta">
                <h2>{item.image_a} vs {item.image_b}</h2>
                <p>Visual score: {item.visual_score:.1f}</p>
                <p>Geometric score: {item.geometric_score:.3f}</p>
                <p>Suspicion: {item.suspicion_score:.1f}</p>
                <p>{item.reason}</p>
              </div>
              <div class="images">
                <img src="{image_a}" alt="{item.image_a}">
                <img src="{image_b}" alt="{item.image_b}">
              </div>
            </article>
            """
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Doppelgangers</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 0; background: #f5f1e8; color: #1d1d1d; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    .pair {{ margin-bottom: 24px; padding: 16px; background: #fffdf8; border: 1px solid #dbd0be; }}
    .images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    img {{ width: 100%; height: 320px; object-fit: cover; background: #e7e0d1; }}
  </style>
</head>
<body>
  <main>
    <h1>Doppelganger Candidates</h1>
    {"".join(cards) if cards else "<p>No suspicious pairs found with the current thresholds.</p>"}
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def parse_points3d(points3d_path: Path) -> tuple[list[float], float]:
    errors: list[float] = []
    track_lengths: list[int] = []
    if not points3d_path.exists():
        return errors, 0.0
    with points3d_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            errors.append(float(parts[7]))
            if len(parts) > 8:
                track_lengths.append(len(parts[8:]) // 2)
    return errors, float(mean(track_lengths)) if track_lengths else 0.0


def count_registered_images(images_txt_path: Path) -> int:
    if not images_txt_path.exists():
        return 0
    count = 0
    with images_txt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "." in stripped.split()[-1]:
                count += 1
    return count


def collect_reconstruction_metrics(model_dir: Path, num_input_images: int) -> dict[str, float | int | bool]:
    points_errors, avg_track_length = parse_points3d(model_dir / "points3D.txt")
    registered_images = count_registered_images(model_dir / "images.txt")
    success = registered_images >= 2 and len(points_errors) > 0
    return {
        "success": success,
        "reproj_mean": float(mean(points_errors)) if points_errors else 0.0,
        "reproj_median": float(median(points_errors)) if points_errors else 0.0,
        "registered_images": registered_images,
        "num_input_images": num_input_images,
        "points3d": len(points_errors),
        "avg_track_length": avg_track_length,
        "registration_ratio": (registered_images / num_input_images) if num_input_images else 0.0,
    }


def append_metrics_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "object_id",
        "run_id",
        "num_images",
        "success",
        "reproj_mean",
        "reproj_median",
        "registered_images",
        "registration_ratio",
        "points3d",
        "avg_track_length",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_metrics(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

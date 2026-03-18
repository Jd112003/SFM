from __future__ import annotations

import html
from collections import defaultdict
from pathlib import Path


def _as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def _as_bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).lower() in {"1", "true", "yes"}


def _series_by_num_images(rows: list[dict[str, str]], key: str) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["num_images"])].append(_as_float(row, key))
    return sorted((num_images, sum(values) / len(values)) for num_images, values in grouped.items())


def write_line_plot_svg(
    points: list[tuple[int, float]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    stroke: str = "#b64926",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 960
    height = 540
    margin_left = 90
    margin_bottom = 70
    margin_top = 60
    margin_right = 40
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    if not points:
        output_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1

    def scale_x(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_width

    def scale_y(value: float) -> float:
        return margin_top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    line_points = " ".join(f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in points)
    dots = "\n".join(
        f"<circle cx='{scale_x(x):.2f}' cy='{scale_y(y):.2f}' r='5' fill='{stroke}' />" for x, y in points
    )
    x_ticks = "\n".join(
        f"<text x='{scale_x(x):.2f}' y='{height - 28}' font-size='14' text-anchor='middle'>{x}</text>"
        for x in xs
    )
    y_ticks_values = [y_min + (i * (y_max - y_min) / 4.0) for i in range(5)]
    y_ticks = "\n".join(
        f"<text x='{margin_left - 12}' y='{scale_y(value):.2f}' font-size='14' text-anchor='end'>{value:.2f}</text>"
        for value in y_ticks_values
    )
    grid = "\n".join(
        f"<line x1='{margin_left}' y1='{scale_y(value):.2f}' x2='{width - margin_right}' y2='{scale_y(value):.2f}' "
        f"stroke='#ddd5c7' stroke-dasharray='4 6' />"
        for value in y_ticks_values
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#f7f4ec" />
  <text x="{width / 2}" y="34" font-size="26" text-anchor="middle" font-family="Georgia">{html.escape(title)}</text>
  {grid}
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#332d24" />
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#332d24" />
  <polyline fill="none" stroke="{stroke}" stroke-width="4" points="{line_points}" />
  {dots}
  {x_ticks}
  {y_ticks}
  <text x="{width / 2}" y="{height - 8}" font-size="16" text-anchor="middle" font-family="Georgia">{html.escape(x_label)}</text>
  <text x="22" y="{height / 2}" font-size="16" text-anchor="middle" transform="rotate(-90 22 {height / 2})" font-family="Georgia">{html.escape(y_label)}</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def write_metrics_plots(rows: list[dict[str, str]], output_dir: Path, include_success_plot: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_points = _series_by_num_images(rows, "reproj_mean")
    median_points = _series_by_num_images(rows, "reproj_median")
    registered_points = _series_by_num_images(rows, "registration_ratio")
    points3d_points = _series_by_num_images(rows, "points3d")

    write_line_plot_svg(
        mean_points,
        output_dir / "error_vs_num_images_mean.svg",
        "Mean Reprojection Error vs Number of Photos",
        "Number of photos",
        "Mean reprojection error",
    )
    write_line_plot_svg(
        median_points,
        output_dir / "error_vs_num_images_median.svg",
        "Median Reprojection Error vs Number of Photos",
        "Number of photos",
        "Median reprojection error",
        stroke="#2b7a78",
    )
    write_line_plot_svg(
        registered_points,
        output_dir / "registration_ratio_vs_num_images.svg",
        "Registration Ratio vs Number of Photos",
        "Number of photos",
        "Registered / input images",
        stroke="#5c4b8a",
    )
    write_line_plot_svg(
        points3d_points,
        output_dir / "points3d_vs_num_images.svg",
        "3D Points vs Number of Photos",
        "Number of photos",
        "3D points",
        stroke="#467143",
    )
    if include_success_plot:
        success_rows = [
            {
                "num_images": row["num_images"],
                "success": "1" if _as_bool(row, "success") else "0",
            }
            for row in rows
        ]
        success_points = _series_by_num_images(success_rows, "success")
        write_line_plot_svg(
            success_points,
            output_dir / "success_rate_vs_num_images.svg",
            "Success Rate vs Number of Photos",
            "Number of photos",
            "Fraction of successful runs",
            stroke="#8d5a2b",
        )

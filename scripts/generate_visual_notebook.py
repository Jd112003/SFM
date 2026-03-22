from __future__ import annotations

import json
from pathlib import Path


def md_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict[str, object]:
    cells = [
        md_cell(
            """# Flujo visual de reconstruccion 3D con SfM

Este notebook convierte el pipeline actual del repositorio en un entregable mas cercano a la consigna original del deber. No vuelve a ejecutar COLMAP; en cambio, **documenta y visualiza cada fase del flujo SfM** a partir de los artefactos ya generados para los tres datasets:

- `buda`
- `estatua`
- `leon`

La idea es usarlo como apoyo del informe: permite mostrar captura, conectividad entre vistas, reconstruccion incremental, comparacion del modelo bruto contra el modelo filtrado y el efecto doppelganger.
"""
        ),
        md_cell(
            """## Alcance del notebook

Este material se apoya en resultados ya producidos por el pipeline de scripts y Docker. Eso permite cumplir con la parte de **presentacion por fases** sin obligar a que el notebook ejecute el 100% de la reconstruccion.

Las fuentes visualizadas aqui son:

- `outputs/<objeto>/metrics/metrics.csv`
- `outputs/<objeto>/graphs/image_graph_edges.csv`
- `outputs/<objeto>/reconstruction/text_model/*.txt`
- `outputs/<objeto>/reconstruction/filtered_text_model/*.txt`
- `outputs/<objeto>/doppelgangers/doppelgangers.csv`
"""
        ),
        code_cell(
            """from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

from sfm_pipeline.notebook_utils import (
    OBJECT_IDS,
    aggregate_metrics,
    build_dataset_summary,
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

plt.style.use("seaborn-v0_8-whitegrid")
ROOT = resolve_repo_root()
OBJECTS = list(OBJECT_IDS)
ROOT
"""
        ),
        md_cell(
            """## Fase 0. Vista general de datasets y resultados finales

Primero resumimos el tamano de cada set, el numero de imagenes efectivamente registradas y la cantidad de puntos 3D obtenidos antes y despues del filtrado.
"""
        ),
        code_cell(
            """summary_df = build_dataset_summary(ROOT, OBJECTS)
summary_df
"""
        ),
        code_cell(
            """fig, axes = plt.subplots(1, len(OBJECTS), figsize=(18, 5))
for ax, object_id in zip(axes, OBJECTS, strict=True):
    sheet = make_contact_sheet(sample_image_paths(ROOT, object_id, max_images=6), cols=3)
    ax.imshow(sheet)
    ax.set_title(f"{object_id}: muestra de vistas")
    ax.axis("off")
fig.suptitle("Muestreo visual de cada dataset", fontsize=16)
plt.tight_layout()
"""
        ),
        md_cell(
            """## Fase 1. Matching y conectividad entre vistas

El pipeline principal usa `sequential_matcher` para la reconstruccion y construye un grafo de relaciones entre imagenes a partir de la base de COLMAP. En las figuras siguientes:

- cada nodo representa una imagen;
- cada arista representa una relacion geometricamente valida;
- el grosor de la arista crece con el `inlier_ratio`.
"""
        ),
        code_cell(
            """fig, axes = plt.subplots(1, len(OBJECTS), figsize=(18, 5))
for ax, object_id in zip(axes, OBJECTS, strict=True):
    nodes, edges = load_graph(ROOT, object_id, min_inliers=15)
    plot_image_graph(ax, nodes, edges, title=f"Grafo de vistas: {object_id}")
fig.tight_layout()
"""
        ),
        code_cell(
            """graph_rows = []
for object_id in OBJECTS:
    nodes, edges = load_graph(ROOT, object_id, min_inliers=15)
    if edges:
        ratios = [edge.inlier_ratio for edge in edges]
        inliers = [edge.inlier_matches for edge in edges]
        graph_rows.append(
            {
                "object_id": object_id,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "mean_inlier_ratio": sum(ratios) / len(ratios),
                "max_inlier_ratio": max(ratios),
                "mean_inlier_matches": sum(inliers) / len(inliers),
            }
        )
graph_df = pd.DataFrame(graph_rows)
graph_df
"""
        ),
        md_cell(
            """## Fase 2. Reconstruccion incremental

La carpeta `metrics/metrics.csv` registra las corridas por subconjuntos de imagenes (`n20`, `n40`, `n80`, etc.). Esto permite mostrar como evoluciona la reconstruccion al aumentar el numero de vistas:

- razon de registro;
- numero de puntos 3D;
- error medio de reproyeccion.
"""
        ),
        code_cell(
            """fig, axes = plt.subplots(len(OBJECTS), 3, figsize=(16, 12), sharex="col")

for row_index, object_id in enumerate(OBJECTS):
    metrics_df = load_metrics_df(ROOT, object_id)
    aggregated = aggregate_metrics(metrics_df)
    full_row = metrics_df.loc[metrics_df["is_full"]].iloc[0]

    axes[row_index, 0].plot(aggregated["num_images"], aggregated["registration_ratio"], marker="o", color="#2b6cb0")
    axes[row_index, 0].fill_between(
        aggregated["num_images"],
        aggregated["registration_ratio"] - aggregated["registration_std"],
        aggregated["registration_ratio"] + aggregated["registration_std"],
        alpha=0.15,
        color="#2b6cb0",
    )
    axes[row_index, 0].scatter([full_row["num_images"]], [full_row["registration_ratio"]], color="#0f172a", s=45)
    axes[row_index, 0].set_ylabel(f"{object_id}\\nregistro")
    axes[row_index, 0].set_ylim(0, 1.05)

    axes[row_index, 1].plot(aggregated["num_images"], aggregated["points3d"], marker="o", color="#2f855a")
    axes[row_index, 1].fill_between(
        aggregated["num_images"],
        aggregated["points3d"] - aggregated["points3d_std"],
        aggregated["points3d"] + aggregated["points3d_std"],
        alpha=0.15,
        color="#2f855a",
    )
    axes[row_index, 1].scatter([full_row["num_images"]], [full_row["points3d"]], color="#0f172a", s=45)

    axes[row_index, 2].plot(aggregated["num_images"], aggregated["reproj_mean"], marker="o", color="#c05621")
    axes[row_index, 2].fill_between(
        aggregated["num_images"],
        aggregated["reproj_mean"] - aggregated["reproj_std"],
        aggregated["reproj_mean"] + aggregated["reproj_std"],
        alpha=0.15,
        color="#c05621",
    )
    axes[row_index, 2].scatter([full_row["num_images"]], [full_row["reproj_mean"]], color="#0f172a", s=45)

axes[0, 0].set_title("Razon de registro")
axes[0, 1].set_title("Puntos 3D")
axes[0, 2].set_title("Error medio de reproyeccion")

for ax in axes[-1, :]:
    ax.set_xlabel("Numero de imagenes")

plt.tight_layout()
"""
        ),
        code_cell(
            """full_metrics = []
for object_id in OBJECTS:
    metrics_df = load_metrics_df(ROOT, object_id)
    full_row = metrics_df.loc[metrics_df["is_full"]].iloc[0]
    full_metrics.append(
        {
            "object_id": object_id,
            "num_images": int(full_row["num_images"]),
            "registered_images": int(full_row["registered_images"]),
            "registration_ratio": float(full_row["registration_ratio"]),
            "points3d": int(full_row["points3d"]),
            "reproj_mean": float(full_row["reproj_mean"]),
            "avg_track_length": float(full_row["avg_track_length"]),
        }
    )
pd.DataFrame(full_metrics)
"""
        ),
        md_cell(
            """## Fase 3. Visualizacion del modelo 3D

Aqui se compara la nube reconstruida original contra la version filtrada. El objetivo del filtrado es limpiar puntos poco confiables y hacer el objeto mas interpretable para la presentacion final.
"""
        ),
        code_cell(
            """fig = plt.figure(figsize=(16, 15))

for row_index, object_id in enumerate(OBJECTS):
    raw_xyz, raw_rgb, _ = load_point_cloud(ROOT, object_id, filtered=False, max_points=14000)
    raw_cameras = load_camera_centers(ROOT, object_id, filtered=False, max_cameras=220)
    filtered_xyz, filtered_rgb, _ = load_point_cloud(ROOT, object_id, filtered=True, max_points=14000)
    filtered_cameras = load_camera_centers(ROOT, object_id, filtered=True, max_cameras=220)

    ax_raw = fig.add_subplot(len(OBJECTS), 2, row_index * 2 + 1, projection="3d")
    plot_point_cloud(ax_raw, raw_xyz, raw_rgb, raw_cameras, title=f"{object_id}: modelo bruto")

    ax_filtered = fig.add_subplot(len(OBJECTS), 2, row_index * 2 + 2, projection="3d")
    plot_point_cloud(ax_filtered, filtered_xyz, filtered_rgb, filtered_cameras, title=f"{object_id}: modelo filtrado")

plt.tight_layout()
"""
        ),
        code_cell(
            """comparison_rows = []
for object_id in OBJECTS:
    raw_xyz, _, raw_error = load_point_cloud(ROOT, object_id, filtered=False, max_points=50000)
    filtered_xyz, _, filtered_error = load_point_cloud(ROOT, object_id, filtered=True, max_points=50000)
    comparison_rows.append(
        {
            "object_id": object_id,
            "raw_points_sampled": len(raw_xyz),
            "filtered_points_sampled": len(filtered_xyz),
            "raw_error_mean_sampled": raw_error.mean() if len(raw_error) else 0.0,
            "filtered_error_mean_sampled": filtered_error.mean() if len(filtered_error) else 0.0,
        }
    )
pd.DataFrame(comparison_rows)
"""
        ),
        md_cell(
            """## Fase 4. Efecto doppelganger

La consigna pide inducir este efecto a proposito. En el pipeline actual se hace con una base auxiliar y `exhaustive_matcher`, buscando pares con:

- muchos matches visuales;
- bajo acuerdo geometrico relativo.

Eso se interpreta como un posible doppelganger: dos vistas que *se parecen* mucho en descriptores, pero cuya verificacion geometrica no es tan consistente.
"""
        ),
        code_cell(
            """for object_id in OBJECTS:
    display(Markdown(f"### {object_id}"))
    doppel_df = load_doppelgangers_df(ROOT, object_id).head(3)
    display(doppel_df[["image_a", "image_b", "visual_score", "geometric_score", "suspicion_score"]])

    fig, axes = plt.subplots(1, len(doppel_df), figsize=(5 * max(len(doppel_df), 1), 4))
    if len(doppel_df) == 1:
        axes = [axes]
    for ax, row in zip(axes, doppel_df.itertuples(index=False), strict=True):
        strip = make_pair_strip(ROOT, object_id, row.image_a, row.image_b)
        ax.imshow(strip)
        ax.set_title(
            f"{row.image_a[-7:-4]} vs {row.image_b[-7:-4]}\\n"
            f"visual={row.visual_score:.0f}, geom={row.geometric_score:.3f}"
        )
        ax.axis("off")
    plt.tight_layout()
"""
        ),
        md_cell(
            """## Fase 5. Trazabilidad con la consigna

La siguiente tabla sirve como guia para el informe final. Resume como este repo puede alinearse con cada literal del enunciado original.
"""
        ),
        code_cell(
            """traceability = pd.DataFrame(
    [
        {
            "literal": "a",
            "que_pide": "Usar un set propio y maximizar calidad",
            "como_lo_cubre_el_repo": "Los tres datasets fueron reconstruidos con COLMAP y hay modelos filtrados para presentacion final.",
            "evidencia": "datasets buda/estatua/leon + outputs/<objeto>/reconstruction/",
        },
        {
            "literal": "b",
            "que_pide": "Presentar resultados por fases con plots",
            "como_lo_cubre_el_repo": "Este notebook muestra muestras de captura, grafo de vistas, curvas incrementales, nube 3D y errores.",
            "evidencia": "notebooks/sfm_visual_workflow.ipynb",
        },
        {
            "literal": "c",
            "que_pide": "Inducir efecto doppelganger",
            "como_lo_cubre_el_repo": "El pipeline usa exhaustive matching auxiliar y guarda candidatos sospechosos.",
            "evidencia": "outputs/<objeto>/doppelgangers/",
        },
        {
            "literal": "d",
            "que_pide": "Flujo organizado y documentado",
            "como_lo_cubre_el_repo": "La logica operativa queda en scripts/CLI y la narrativa academica en este notebook.",
            "evidencia": "sfm_pipeline/*.py + notebooks/sfm_visual_workflow.ipynb",
        },
        {
            "literal": "e",
            "que_pide": "Informe coherente con analisis",
            "como_lo_cubre_el_repo": "El notebook y las tablas resumen pueden alimentar directamente las secciones del reporte IEEE.",
            "evidencia": "figuras y tablas generadas por este notebook",
        },
        {
            "literal": "f",
            "que_pide": "Subir el dataset empleado",
            "como_lo_cubre_el_repo": "Los datasets estan versionados en carpetas separadas por objeto.",
            "evidencia": "buda/, estatua/, leon/",
        },
    ]
)
traceability
"""
        ),
        md_cell(
            """## Recomendaciones para el informe

1. Explicar que el pipeline reproducible (`scripts + Docker`) fue la herramienta de trabajo, pero que el **notebook es el entregable pedagogico** que evidencia el flujo por fases.
2. Insertar en el reporte una figura de cada bloque: muestra de vistas, grafo de matching, curvas incrementales, nube filtrada y ejemplos de doppelganger.
3. Aclarar que la reconstruccion incremental se obtuvo por ablacion de subconjuntos de imagenes, lo que reemplaza de forma razonable una reconstruccion paso a paso dentro del notebook.
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    output_path = Path(__file__).resolve().parents[1] / "notebooks" / "sfm_visual_workflow.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from image_lab3.io.config_loader import load_config
from image_lab3.report.exporters import export_json
from image_lab3.report.plotting import create_distribution_figure
from image_lab3.services.experiment_runner import ExperimentRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report"
GENERATED_DIR = REPORT_DIR / "generated"


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(PROJECT_ROOT / "examples" / "default_config.json")
    result = ExperimentRunner().run(config)
    export_json(result, GENERATED_DIR / "results.json")
    _write_distribution_figures(result)
    _write_typst_values(config, result, GENERATED_DIR / "values.typ")


def _write_distribution_figures(result) -> None:
    for key, distribution in result.distributions.items():
        figure = create_distribution_figure(distribution)
        figure.savefig(GENERATED_DIR / f"{key}.png", dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())


def _write_typst_values(config, result, destination: Path) -> None:
    metrics_rows = []
    for distribution in result.distributions.values():
        metrics = distribution.metrics
        metrics_rows.append(
            "  [{title}], [{inside:.6f}], [{constraint:.6e}], [{mean:.6e}], [{uniformity:.6f}],".format(
                title=distribution.title,
                inside=metrics.inside_ratio,
                constraint=metrics.norm_error,
                mean=metrics.centroid_or_mean_error,
                uniformity=metrics.uniformity_score,
            )
        )

    triangle_rectangles = _triangle_rectangles_table(result.distributions["triangle"].auxiliary)
    circle_rectangles = _circle_rectangles_table(result.distributions["circle"].auxiliary)

    content = """#let sample_count_label = "{sample_count}"
#let uniformity_rectangle_count = "{uniformity_rectangle_count}"
#let triangle_a = "({triangle_ax:.3f}, {triangle_ay:.3f}, {triangle_az:.3f})"
#let triangle_b = "({triangle_bx:.3f}, {triangle_by:.3f}, {triangle_bz:.3f})"
#let triangle_c = "({triangle_cx:.3f}, {triangle_cy:.3f}, {triangle_cz:.3f})"
#let circle_center = "({circle_cx:.3f}, {circle_cy:.3f}, {circle_cz:.3f})"
#let circle_normal = "({circle_nx:.3f}, {circle_ny:.3f}, {circle_nz:.3f})"
#let circle_radius = "{circle_radius:.3f}"
#let cosine_normal = "({cosine_nx:.3f}, {cosine_ny:.3f}, {cosine_nz:.3f})"

#let triangle_inside = "{triangle_inside:.6f}"
#let triangle_constraint = "{triangle_constraint:.6e}"
#let triangle_mean = "{triangle_mean:.6e}"
#let triangle_uniformity = "{triangle_uniformity:.6f}"

#let circle_inside = "{circle_inside:.6f}"
#let circle_constraint = "{circle_constraint:.6e}"
#let circle_mean = "{circle_mean:.6e}"
#let circle_uniformity = "{circle_uniformity:.6f}"

#let sphere_inside = "{sphere_inside:.6f}"
#let sphere_constraint = "{sphere_constraint:.6e}"
#let sphere_mean = "{sphere_mean:.6e}"
#let sphere_uniformity = "{sphere_uniformity:.6f}"

#let cosine_inside = "{cosine_inside:.6f}"
#let cosine_constraint = "{cosine_constraint:.6e}"
#let cosine_mean = "{cosine_mean:.6e}"
#let cosine_uniformity = "{cosine_uniformity:.6f}"

#let metrics_table = table(
  columns: 5,
  inset: 6pt,
  stroke: (x, y) => if y == 0 {{ 0.9pt + rgb("#3f5974") }} else {{ 0.5pt + rgb("#8aa0b5") }},
  align: center,
  [Распределение], [Inside ratio], [Constraint error], [Mean error], [Uniformity score],
{metrics_rows}
)

#let triangle_rectangles_table = table(
  columns: 3,
  inset: 6pt,
  stroke: (x, y) => if y == 0 {{ 0.9pt + rgb("#3f5974") }} else {{ 0.5pt + rgb("#8aa0b5") }},
  align: center,
  [Прямоугольник], [Диапазон в (u, v)], [Число точек],
{triangle_rectangles}
)

#let circle_rectangles_table = table(
  columns: 3,
  inset: 6pt,
  stroke: (x, y) => if y == 0 {{ 0.9pt + rgb("#3f5974") }} else {{ 0.5pt + rgb("#8aa0b5") }},
  align: center,
  [Прямоугольник], [Диапазон в локальных (x, y)], [Число точек],
{circle_rectangles}
)
    """.format(
        sample_count=result.sample_count,
        uniformity_rectangle_count=config.uniformity_rectangle_count,
        triangle_ax=config.triangle.a.x,
        triangle_ay=config.triangle.a.y,
        triangle_az=config.triangle.a.z,
        triangle_bx=config.triangle.b.x,
        triangle_by=config.triangle.b.y,
        triangle_bz=config.triangle.b.z,
        triangle_cx=config.triangle.c.x,
        triangle_cy=config.triangle.c.y,
        triangle_cz=config.triangle.c.z,
        circle_cx=config.circle.center.x,
        circle_cy=config.circle.center.y,
        circle_cz=config.circle.center.z,
        circle_nx=config.circle.normal.x,
        circle_ny=config.circle.normal.y,
        circle_nz=config.circle.normal.z,
        circle_radius=config.circle.radius,
        cosine_nx=config.cosine_normal.x,
        cosine_ny=config.cosine_normal.y,
        cosine_nz=config.cosine_normal.z,
        triangle_inside=result.distributions["triangle"].metrics.inside_ratio,
        triangle_constraint=result.distributions["triangle"].metrics.norm_error,
        triangle_mean=result.distributions["triangle"].metrics.centroid_or_mean_error,
        triangle_uniformity=result.distributions["triangle"].metrics.uniformity_score,
        circle_inside=result.distributions["circle"].metrics.inside_ratio,
        circle_constraint=result.distributions["circle"].metrics.norm_error,
        circle_mean=result.distributions["circle"].metrics.centroid_or_mean_error,
        circle_uniformity=result.distributions["circle"].metrics.uniformity_score,
        sphere_inside=result.distributions["sphere"].metrics.inside_ratio,
        sphere_constraint=result.distributions["sphere"].metrics.norm_error,
        sphere_mean=result.distributions["sphere"].metrics.centroid_or_mean_error,
        sphere_uniformity=result.distributions["sphere"].metrics.uniformity_score,
        cosine_inside=result.distributions["cosine"].metrics.inside_ratio,
        cosine_constraint=result.distributions["cosine"].metrics.norm_error,
        cosine_mean=result.distributions["cosine"].metrics.centroid_or_mean_error,
        cosine_uniformity=result.distributions["cosine"].metrics.uniformity_score,
        metrics_rows="\n".join(metrics_rows),
        triangle_rectangles=triangle_rectangles,
        circle_rectangles=circle_rectangles,
    )
    destination.write_text(content, encoding="utf-8")


def _triangle_rectangles_table(auxiliary) -> str:
    u = auxiliary["beta"]
    v = auxiliary["gamma"]
    rectangles = auxiliary["proof_rectangles"]
    return _format_rectangle_rows(
        u,
        v,
        rectangles,
        x_label="u",
        y_label="v",
    )


def _circle_rectangles_table(auxiliary) -> str:
    local_x = auxiliary["local_x"]
    local_y = auxiliary["local_y"]
    rectangles = auxiliary["proof_rectangles"]
    return _format_rectangle_rows(
        local_x,
        local_y,
        rectangles,
        x_label="x",
        y_label="y",
    )


def _format_rectangle_rows(values_x, values_y, rectangles, x_label: str, y_label: str) -> str:
    rows = []
    for name, x0, x1, y0, y1 in rectangles:
        count = int(np.sum((values_x >= x0) & (values_x <= x1) & (values_y >= y0) & (values_y <= y1)))
        rows.append(
            "  [{name}], [{x_label}∈[{x0:.2f}; {x1:.2f}], {y_label}∈[{y0:.2f}; {y1:.2f}]], [{count}],".format(
                name=name,
                x_label=x_label,
                x0=x0,
                x1=x1,
                y_label=y_label,
                y0=y0,
                y1=y1,
                count=count,
            )
        )
    return "\n".join(rows)


if __name__ == "__main__":
    main()

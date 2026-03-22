from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from image_lab1.io.scenario_loader import load_scenario
from image_lab1.report.exporters import export_json
from image_lab1.services.observer_renderer import ObserverRenderer
from image_lab1.services.scenario_runner import ScenarioRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report"
GENERATED_DIR = REPORT_DIR / "generated"
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario(EXAMPLES_DIR / "demo_scenario.json")
    result = ScenarioRunner().run(scenario)

    export_json(result, GENERATED_DIR / "results.json")
    _write_scene_image(scenario, result, GENERATED_DIR / "scene_3d.png")
    _write_observer_render(scenario, GENERATED_DIR / "observer_render.png")
    _write_typst_values(scenario, result, GENERATED_DIR / "values.typ")


def _write_scene_image(scenario, result, destination: Path) -> None:
    figure = Figure(figsize=(8, 6), facecolor="#0f1520")
    axes = figure.add_subplot(111, projection="3d")
    axes.set_facecolor("#0f1520")

    triangle = scenario.triangle
    face_color = (0.18, 0.43, 0.68, 0.45)
    triangle_surface = Poly3DCollection(
        [[triangle.a.to_tuple(), triangle.b.to_tuple(), triangle.c.to_tuple()]],
        facecolors=[face_color],
        edgecolors="#71d8ff",
        linewidths=2.0,
    )
    axes.add_collection3d(triangle_surface)
    for point_name, point in (("A", triangle.a), ("B", triangle.b), ("C", triangle.c)):
        axes.scatter([point.x], [point.y], [point.z], color="#71d8ff", s=40)
        axes.text(point.x, point.y, point.z, " " + point_name, color="white")

    for index, light in enumerate(scenario.lights, start=1):
        axes.scatter(
            [light.position.x],
            [light.position.y],
            [light.position.z],
            color=(light.intensity.r, light.intensity.g, light.intensity.b),
            s=120,
            marker="*",
        )
        direction = light.axis_direction.normalized() * 0.8
        axes.quiver(
            light.position.x,
            light.position.y,
            light.position.z,
            direction.x,
            direction.y,
            direction.z,
            color="#ffd36b",
        )
        axes.text(light.position.x, light.position.y, light.position.z, f" L{index}", color="#ffeab5")

    observer = scenario.observer.position
    axes.scatter([observer.x], [observer.y], [observer.z], color="#ff5b9a", s=65, marker="s")
    axes.text(observer.x, observer.y, observer.z, " Camera", color="#ffb1cb")

    for item in result.local_evaluations:
        color = item.total_brightness.clamp()
        axes.scatter(
            [item.global_point.x],
            [item.global_point.y],
            [item.global_point.z],
            color=(color.r, color.g, color.b),
            s=55,
        )
        normal = item.normal * 0.45
        axes.quiver(
            item.global_point.x,
            item.global_point.y,
            item.global_point.z,
            normal.x,
            normal.y,
            normal.z,
            color="#97d85f",
        )
        axes.text(item.global_point.x, item.global_point.y, item.global_point.z, f" {item.point_name}", color="white")

    axes.set_xlabel("X", color="white")
    axes.set_ylabel("Y", color="white")
    axes.set_zlabel("Z", color="white")
    axes.tick_params(colors="white")
    axes.set_title("3D-схема сцены", color="white")
    _set_equal_limits(axes, scenario, result)
    axes.view_init(elev=24, azim=-68)
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())


def _write_observer_render(scenario, destination: Path) -> None:
    render = ObserverRenderer(width=900, height=900, fov_degrees=34.0).render(scenario)
    Image.fromarray(render.image, mode="RGB").save(destination)


def _write_typst_values(scenario, result, destination: Path) -> None:
    local_rows = _table_rows(
        result.local_evaluations,
        title_mode="local",
    )
    global_rows = _table_rows(
        result.global_evaluations,
        title_mode="global",
    )
    contribution_rows = _contribution_rows(result.local_evaluations)

    content = f"""#let student_name = "Студент"
#let report_date = "22.03.2026"
#let triangle_a = "({scenario.triangle.a.x:.3f}, {scenario.triangle.a.y:.3f}, {scenario.triangle.a.z:.3f})"
#let triangle_b = "({scenario.triangle.b.x:.3f}, {scenario.triangle.b.y:.3f}, {scenario.triangle.b.z:.3f})"
#let triangle_c = "({scenario.triangle.c.x:.3f}, {scenario.triangle.c.y:.3f}, {scenario.triangle.c.z:.3f})"
#let observer_position = "({scenario.observer.position.x:.3f}, {scenario.observer.position.y:.3f}, {scenario.observer.position.z:.3f})"
#let material_color = "({scenario.material.color.r:.2f}, {scenario.material.color.g:.2f}, {scenario.material.color.b:.2f})"
#let material_kd = "{scenario.material.diffuse_coefficient:.2f}"
#let material_ks = "{scenario.material.specular_coefficient:.2f}"
#let material_shininess = "{scenario.material.shininess:.1f}"
#let light_1 = "{scenario.lights[0].name}: pos=({scenario.lights[0].position.x:.2f}, {scenario.lights[0].position.y:.2f}, {scenario.lights[0].position.z:.2f}), axis=({scenario.lights[0].axis_direction.x:.2f}, {scenario.lights[0].axis_direction.y:.2f}, {scenario.lights[0].axis_direction.z:.2f}), I=({scenario.lights[0].intensity.r:.2f}, {scenario.lights[0].intensity.g:.2f}, {scenario.lights[0].intensity.b:.2f})"
#let light_2 = "{scenario.lights[1].name}: pos=({scenario.lights[1].position.x:.2f}, {scenario.lights[1].position.y:.2f}, {scenario.lights[1].position.z:.2f}), axis=({scenario.lights[1].axis_direction.x:.2f}, {scenario.lights[1].axis_direction.y:.2f}, {scenario.lights[1].axis_direction.z:.2f}), I=({scenario.lights[1].intensity.r:.2f}, {scenario.lights[1].intensity.g:.2f}, {scenario.lights[1].intensity.b:.2f})"
#let normal_value = "({result.local_evaluations[0].normal.x:.4f}, {result.local_evaluations[0].normal.y:.4f}, {result.local_evaluations[0].normal.z:.4f})"

#let local_points_table = table(
  columns: 5,
  inset: 6pt,
  stroke: (x, y) => if y == 0 {{ 0.9pt + rgb("#3f5974") }} else {{ 0.5pt + rgb("#8aa0b5") }},
  align: center,
  [Точка], [Локальные координаты], [Глобальные координаты], [Освещенность RGB], [Яркость RGB],
{local_rows}
)

#let global_points_table = table(
  columns: 5,
  inset: 6pt,
  stroke: (x, y) => if y == 0 {{ 0.9pt + rgb("#3f5974") }} else {{ 0.5pt + rgb("#8aa0b5") }},
  align: center,
  [Точка], [Локальные координаты], [Глобальные координаты], [Освещенность RGB], [Яркость RGB],
{global_rows}
)

#let source_contributions_table = table(
  columns: 7,
  inset: 6pt,
  stroke: (x, y) => if y == 0 {{ 0.9pt + rgb("#3f5974") }} else {{ 0.5pt + rgb("#8aa0b5") }},
  align: center,
  [Точка], [Источник], [$theta$, °], [$alpha$, °], [Профиль], [BRDF], [Яркость RGB],
{contribution_rows}
)
"""
    destination.write_text(content, encoding="utf-8")


def _table_rows(evaluations, title_mode: str) -> str:
    rows = []
    for item in evaluations:
        rows.append(
            "  [{name}], [{local}], [{global_}], [{irr}], [{bright}],".format(
                name=item.point_name,
                local="({:.3f}, {:.3f})".format(item.local_coordinates[0], item.local_coordinates[1]),
                global_="({:.3f}, {:.3f}, {:.3f})".format(
                    item.global_point.x, item.global_point.y, item.global_point.z
                ),
                irr="({:.5f}, {:.5f}, {:.5f})".format(
                    item.total_irradiance.r, item.total_irradiance.g, item.total_irradiance.b
                ),
                bright="({:.5f}, {:.5f}, {:.5f})".format(
                    item.total_brightness.r, item.total_brightness.g, item.total_brightness.b
                ),
            )
        )
    return "\n".join(rows)


def _contribution_rows(evaluations) -> str:
    rows = []
    for item in evaluations:
        for contribution in item.contributions:
            rows.append(
                "  [{point}], [{light}], [{theta:.2f}], [{alpha:.2f}], [{profile:.5f}], [{brdf:.5f}], [{brightness}],".format(
                    point=item.point_name,
                    light=contribution.light_name,
                    theta=contribution.theta_degrees,
                    alpha=contribution.alpha_degrees,
                    profile=contribution.angular_profile,
                    brdf=contribution.brdf_value,
                    brightness="({:.5f}, {:.5f}, {:.5f})".format(
                        contribution.brightness.r,
                        contribution.brightness.g,
                        contribution.brightness.b,
                    ),
                )
            )
    return "\n".join(rows)


def _set_equal_limits(axes, scenario, result) -> None:
    points = [
        scenario.triangle.a,
        scenario.triangle.b,
        scenario.triangle.c,
        scenario.observer.position,
        *[light.position for light in scenario.lights],
        *[item.global_point for item in result.local_evaluations],
    ]
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    zs = [point.z for point in points]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    center_z = (min(zs) + max(zs)) / 2.0
    radius = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0) / 2.0 + 0.5
    axes.set_xlim(center_x - radius, center_x + radius)
    axes.set_ylim(center_y - radius, center_y + radius)
    axes.set_zlim(center_z - radius, center_z + radius)


if __name__ == "__main__":
    main()

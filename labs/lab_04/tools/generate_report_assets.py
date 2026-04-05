from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle

from image_lab4.io.config_loader import load_config
from image_lab4.report.exporters import save_hdr, save_png, save_ppm
from image_lab4.services.path_tracer import PathTracer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report"
GENERATED_DIR = REPORT_DIR / "generated"


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(PROJECT_ROOT / "examples" / "default_scene.json")
    _write_scene_plan(config, GENERATED_DIR / "scene_plan.png")

    preview_png = GENERATED_DIR / "preview.png"
    preview_hdr = GENERATED_DIR / "preview.hdr"
    if not preview_png.exists() or not preview_hdr.exists():
        tracer = PathTracer()
        preview_config = _with_render_overrides(config, width=256, height=256, samples_per_pixel=1, max_depth=2, min_depth=1)
        artifact = tracer.render(preview_config, strict_resolution=False)
        save_ppm(GENERATED_DIR / "preview.ppm", artifact.display)
        save_png(preview_png, artifact.display)
        save_hdr(preview_hdr, artifact.radiance)
        scene = artifact.scene
        max_radiance = float(np.max(artifact.radiance))
        mean_radiance = float(np.mean(artifact.radiance))
    else:
        scene = None
        max_radiance = "сохраняется в HDR и анализируется отдельно"
        mean_radiance = "определяется текущими параметрами рендера"

    _write_hdr_comparison(preview_png, preview_hdr, GENERATED_DIR / "hdr_comparison.png")

    triangle_count = len(config.triangles)
    light_count = sum(1 for triangle in config.triangles if max(triangle.emission.to_tuple()) > 0.0)
    if scene is not None:
        triangle_count = len(scene.triangles)
        light_count = len(scene.lights)

    payload = {
        "triangle_count": triangle_count,
        "light_count": light_count,
        "max_radiance": max_radiance,
        "mean_radiance": mean_radiance,
        "reference_spp": config.render.samples_per_pixel,
        "convergence": [],
    }
    (GENERATED_DIR / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_values_typ(config, payload, GENERATED_DIR / "values.typ")


def _with_render_overrides(config, *, width: int, height: int, samples_per_pixel: int, max_depth: int, min_depth: int):
    render = config.render.__class__(
        width=width,
        height=height,
        samples_per_pixel=samples_per_pixel,
        max_depth=max_depth,
        min_depth=min_depth,
        gamma=config.render.gamma,
        normalization=config.render.normalization,
        normalization_value=config.render.normalization_value,
        seed=config.render.seed + samples_per_pixel,
        background=config.render.background,
    )
    return config.__class__(
        camera=config.camera,
        render=render,
        materials=config.materials,
        triangles=config.triangles,
        obj_meshes=config.obj_meshes,
    )


def _write_scene_plan(config, destination: Path) -> None:
    figure = Figure(figsize=(8.8, 6.1), facecolor="#f5efe5")
    axes = figure.add_subplot(111)
    axes.set_facecolor("#fffaf1")

    room = _bounds(config.triangles[0:10])
    _draw_bbox(axes, room, facecolor="#f3eee5", edgecolor="#85654f", linewidth=2.0, label="Комната")

    left_wall = _bounds(config.triangles[6:8])
    right_wall = _bounds(config.triangles[8:10])
    _draw_bbox(axes, left_wall, facecolor="#db7468", edgecolor="#8a3e3a", linewidth=2.0, alpha=0.45, label="Красная стена")
    _draw_bbox(axes, right_wall, facecolor="#79c275", edgecolor="#3e7a42", linewidth=2.0, alpha=0.45, label="Зелёная стена")

    objects = [
        ("Низкая тумба", _bounds(config.triangles[18:28]), "#ba8a5a"),
        ("Синяя коробка", _bounds(config.triangles[28:38]), "#6376e7"),
        ("Белый подиум", _bounds(config.triangles[38:48]), "#d7d6d1"),
        ("Зеркальная панель", _bounds(config.triangles[48:50]), "#444444"),
        ("Фиолетовый шкаф", _bounds(config.triangles[50:60]), "#9d64ce"),
        ("Пирамида 1", _obj_bbox(config.obj_meshes[0]), "#222222"),
        ("Пирамида 2", _obj_bbox(config.obj_meshes[1]), "#d1a265"),
    ]
    for label, bbox, color in objects:
        _draw_bbox(axes, bbox, facecolor=color, edgecolor="#27313f", linewidth=1.3, alpha=0.88, label=label)

    for index, triangles in enumerate((config.triangles[10:12], config.triangles[12:14], config.triangles[14:16], config.triangles[16:18]), start=1):
        center = _centroid(triangles)
        axes.scatter(center[0], center[2], s=90, marker="*", color="#f3a22b", edgecolors="#6d4306", linewidths=0.8, zorder=6)
        axes.text(center[0] + 0.03, center[2] + 0.03, f"L{index}", fontsize=10, color="#704200")

    camera = config.camera.position
    target = config.camera.target
    axes.scatter(camera.x, camera.z, s=120, color="#ca5f3e", edgecolors="white", linewidths=1.2, zorder=7)
    axes.text(camera.x + 0.08, camera.z + 0.05, "Камера", fontsize=11, weight="bold", color="#8a341f")
    axes.add_patch(
        FancyArrowPatch(
            (camera.x, camera.z),
            (target.x, target.z),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color="#ca5f3e",
        )
    )

    axes.set_title("Схема текущей комнаты сверху", fontsize=16, weight="bold", color="#243040")
    axes.set_xlabel("X")
    axes.set_ylabel("Z")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(alpha=0.18)
    axes.set_xlim(room[0] - 0.25, room[1] + 0.25)
    axes.set_ylim(room[2] - 0.25, room[3] + 0.25)
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())


def _write_hdr_comparison(ldr_path: Path, hdr_path: Path, destination: Path) -> None:
    figure = Figure(figsize=(10.2, 4.8), facecolor="#f5efe5")
    left = figure.add_subplot(121)
    right = figure.add_subplot(122)

    ldr = cv2.cvtColor(cv2.imread(str(ldr_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    hdr = cv2.cvtColor(cv2.imread(str(hdr_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    hdr = np.clip(hdr, 0.0, None)
    hdr_visual = _tonemap_hdr_for_report(hdr, exposure=1.75)

    for axes, image, title in (
        (left, ldr, "Обычный PNG после нормировки и gamma"),
        (right, hdr_visual, "HDR-визуализация с повышенной экспозицией"),
    ):
        axes.imshow(image)
        axes.set_title(title, fontsize=12, color="#243040")
        axes.axis("off")

    figure.suptitle("Сравнение LDR и HDR-представления одной сцены", fontsize=15, weight="bold", color="#243040")
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())


def _bounds(triangles) -> Tuple[float, float, float, float]:
    xs = []
    zs = []
    for triangle in triangles:
        xs.extend((triangle.a.x, triangle.b.x, triangle.c.x))
        zs.extend((triangle.a.z, triangle.b.z, triangle.c.z))
    return min(xs), max(xs), min(zs), max(zs)


def _centroid(triangles) -> Tuple[float, float, float]:
    xs = []
    ys = []
    zs = []
    for triangle in triangles:
        xs.extend((triangle.a.x, triangle.b.x, triangle.c.x))
        ys.extend((triangle.a.y, triangle.b.y, triangle.c.y))
        zs.extend((triangle.a.z, triangle.b.z, triangle.c.z))
    return float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))


def _obj_bbox(mesh) -> Tuple[float, float, float, float]:
    size = mesh.transform.scale
    x = mesh.transform.translate.x
    z = mesh.transform.translate.z
    half = size * 0.52
    return x - half, x + half, z - half, z + half


def _draw_bbox(axes, bbox, *, facecolor: str, edgecolor: str, linewidth: float, label: str, alpha: float = 0.28) -> None:
    min_x, max_x, min_z, max_z = bbox
    rect = Rectangle(
        (min_x, min_z),
        max_x - min_x,
        max_z - min_z,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )
    axes.add_patch(rect)
    axes.text(
        (min_x + max_x) * 0.5,
        (min_z + max_z) * 0.5,
        label,
        ha="center",
        va="center",
        fontsize=9,
        color="#15212d",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#fff9f0", "edgecolor": "none", "alpha": 0.8},
    )


def _tonemap_hdr_for_report(hdr: np.ndarray, exposure: float) -> np.ndarray:
    boosted = hdr * exposure
    percentile = float(np.percentile(boosted, 99.4))
    scale = percentile if percentile > 1e-6 else 1.0
    mapped = np.clip(boosted / scale, 0.0, 1.0)
    return np.power(mapped, 1.0 / 2.2)


def _write_convergence_plot(points, destination: Path) -> None:
    figure = Figure(figsize=(8, 4.5), facecolor="#f5efe5")
    axes = figure.add_subplot(111)
    axes.set_facecolor("#fffaf1")
    axes.plot([item[0] for item in points], [item[1] for item in points], marker="o", color="#ca5f3e")
    axes.set_xlabel("Samples per pixel")
    axes.set_ylabel("MSE к эталону")
    axes.set_title("Сходимость path tracing по числу лучей на пиксель")
    axes.grid(alpha=0.3)
    figure.savefig(destination, dpi=170, bbox_inches="tight")


def _write_values_typ(config, payload, destination: Path) -> None:
    convergence_rows = []
    for item in payload["convergence"]:
        convergence_rows.append(
            "  [{spp}], [{mse:.8e}], [{mean:.8e}],".format(
                spp=item["spp"],
                mse=item["mse_to_reference"],
                mean=item["mean_luminance"],
            )
        )
    max_radiance = payload["max_radiance"]
    mean_radiance = payload["mean_radiance"]
    if isinstance(max_radiance, str):
        max_radiance = 0.0
    if isinstance(mean_radiance, str):
        mean_radiance = 0.0
    content = """#let image_width = "{width}"
#let image_height = "{height}"
#let spp_reference = "{reference_spp}"
#let max_depth = "{max_depth}"
#let gamma_value = "{gamma:.2f}"
#let normalization_mode = "{normalization}"
#let triangle_count = "{triangle_count}"
#let light_count = "{light_count}"
#let max_radiance = "{max_radiance:.8f}"
#let mean_radiance = "{mean_radiance:.8f}"

#let convergence_table = table(
  columns: 3,
  inset: 6pt,
  stroke: (x, y) => if y == 0 {{ 0.9pt + rgb("#3f5974") }} else {{ 0.5pt + rgb("#8aa0b5") }},
  align: center,
  [SPP], [MSE к эталону], [Средняя яркость],
{rows}
)
""".format(
        width=config.render.width,
        height=config.render.height,
        reference_spp=payload["reference_spp"],
        max_depth=config.render.max_depth,
        gamma=config.render.gamma,
        normalization=config.render.normalization,
        triangle_count=payload["triangle_count"],
        light_count=payload["light_count"],
        max_radiance=max_radiance,
        mean_radiance=mean_radiance,
        rows="\n".join(convergence_rows),
    )
    destination.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()

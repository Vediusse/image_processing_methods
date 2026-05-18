from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from image_lab4.io.config_loader import load_config
from image_lab4.report.exporters import save_png
from image_lab4.services.path_tracer import PathTracer
from image_lab4.services.taichi_progressive_path_tracer import TaichiProgressivePathTracer


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"
CONFIG = ROOT / "examples" / "default_scene.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-renders", action="store_true", help="Render SPP comparison through Taichi/Metal.")
    args = parser.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG)
    scene = PathTracer()._build_scene(config, strict_resolution=False)

    draw_scene_map(config, scene)
    draw_pixel_pipeline()
    draw_light_importance(scene)
    draw_path_example(config, scene)
    draw_material_model(config)
    if args.with_renders:
        draw_accumulation_examples(config)
    else:
        draw_accumulation_concept()


def draw_scene_map(config, scene) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#f7f1e7")
    ax.set_facecolor("#fffaf2")
    ax.set_title("Сцена сверху: камера, объекты, area lights и point light", fontsize=17, weight="bold")

    floor = _bounds([scene.triangles[0], scene.triangles[1]])
    _rect(ax, floor, "#efe5d3", "#8b6b46", "расширенный пол")

    material_colors = {
        "wood": "#b48355",
        "blue_box": "#3b62d6",
        "white": "#d8d4c8",
        "mirror": "#3ab75a",
        "violet": "#a85bd0",
        "metal_dark": "#1d1d1f",
    }
    grouped = {}
    for tri in scene.triangles[10:]:
        grouped.setdefault(tri.material.name, []).append(tri)
    for name, triangles in grouped.items():
        if name == "white" and len(triangles) < 4:
            continue
        bbox = _bounds(triangles)
        label = {
            "wood": "деревянная тумба",
            "blue_box": "синий куб",
            "white": "белый подиум",
            "mirror": "зеленое зеркало",
            "violet": "фиолетовый шкаф",
            "metal_dark": "темная пирамида",
        }.get(name, name)
        _rect(ax, bbox, material_colors.get(name, "#cccccc"), "#1f2a34", label, alpha=0.82)

    for index, light in enumerate(scene.lights, start=1):
        center = _center(light)
        ax.scatter(center[0], center[2], s=140, marker="*", color="#ffcf4d", edgecolors="#6b4300", zorder=6)
        ax.text(center[0] + 0.05, center[2] + 0.05, f"A{index}", fontsize=10, weight="bold")

    for index, light in enumerate(scene.point_lights, start=1):
        ax.scatter(light.position.x, light.position.z, s=170, marker="o", color="#f9a63a", edgecolors="#401900", zorder=7)
        ax.text(light.position.x + 0.06, light.position.z - 0.06, f"P{index}: point light", fontsize=11, weight="bold")

    camera = config.camera.position
    target = config.camera.target
    ax.scatter(camera.x, camera.z, s=150, color="#e4573d", edgecolors="white", zorder=8)
    ax.text(camera.x + 0.12, camera.z, "камера", fontsize=12, weight="bold")
    ax.add_patch(FancyArrowPatch((camera.x, camera.z), (target.x, target.z), arrowstyle="-|>", mutation_scale=18, linewidth=2.4, color="#e4573d"))

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "01_scene_map.png", dpi=180)
    plt.close(fig)


def draw_pixel_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#101720")
    ax.set_axis_off()
    ax.set_title("Как рождается цвет одного пикселя", color="white", fontsize=20, weight="bold", pad=16)

    boxes = [
        ("Пиксель (x, y)", "Координата пикселя\n+ jitter внутри пикселя", (0.04, 0.60), "#2f80ed"),
        ("Camera ray", "Из камеры через экран\nстроится первый луч", (0.28, 0.60), "#56ccf2"),
        ("Пересечение", "Moller-Trumbore ищет\nближайший треугольник", (0.52, 0.60), "#27ae60"),
        ("Материал", "diffuse / mirror решают:\nрассеять или отразить", (0.76, 0.60), "#f2c94c"),
        ("Свет", "Источник выбирается\nпо значимости", (0.18, 0.24), "#f2994a"),
        ("Вклад", "radiance += light\n/pdf, distance^2, cos", (0.43, 0.24), "#eb5757"),
        ("Накопление", "frame_radiance -> accumulation\naverage -> tone map -> PNG", (0.68, 0.24), "#bb6bd9"),
    ]
    for title, body, (x, y), color in boxes:
        _box(ax, x, y, 0.20, 0.20, title, body, color)

    arrows = [
        ((0.24, 0.70), (0.28, 0.70)),
        ((0.48, 0.70), (0.52, 0.70)),
        ((0.72, 0.70), (0.76, 0.70)),
        ((0.86, 0.60), (0.28, 0.34)),
        ((0.38, 0.34), (0.43, 0.34)),
        ((0.63, 0.34), (0.68, 0.34)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=18, linewidth=2.0, color="#d7e3f4"))

    ax.text(0.5, 0.06, "Главная идея: один путь шумный, много независимых путей дают среднее значение пикселя.", ha="center", color="#d7e3f4", fontsize=14)
    fig.tight_layout()
    fig.savefig(ASSETS / "02_pixel_pipeline.png", dpi=180)
    plt.close(fig)


def draw_light_importance(scene) -> None:
    labels = [f"A{i + 1}" for i in range(len(scene.lights))] + [f"P{i + 1}" for i in range(len(scene.point_lights))]
    probabilities = scene.light_probabilities
    colors = ["#f2c94c"] * len(scene.lights) + ["#f2994a"] * len(scene.point_lights)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#f7f1e7")
    bars = ax.bar(labels, probabilities, color=colors, edgecolor="#34281d")
    ax.set_title("Выбор источника света по значимости", fontsize=18, weight="bold")
    ax.set_ylabel("Вероятность выбора")
    ax.set_ylim(0.0, max(probabilities) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    for bar, probability in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01, f"{probability:.2f}", ha="center", fontsize=10)
    ax.text(
        0.5,
        -0.18,
        "Area light: вес = площадь * средняя emission. Point light: вес = средняя intensity, поэтому нулевая площадь не убивает вклад.",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "03_light_importance.png", dpi=180)
    plt.close(fig)


def draw_path_example(config, scene) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#f7f1e7")
    ax.set_facecolor("#fffaf2")
    ax.set_title("Пример одного Monte-Carlo пути", fontsize=18, weight="bold")

    floor = _bounds([scene.triangles[0], scene.triangles[1]])
    _rect(ax, floor, "#efe5d3", "#8b6b46", "пол")
    mirror = [tri for tri in scene.triangles if tri.material.name == "mirror"]
    blue = [tri for tri in scene.triangles if tri.material.name == "blue_box"]
    violet = [tri for tri in scene.triangles if tri.material.name == "violet"]
    _rect(ax, _bounds(mirror), "#42bf66", "#134b25", "зеркало")
    _rect(ax, _bounds(blue), "#3b62d6", "#1d2d78", "синий куб")
    _rect(ax, _bounds(violet), "#a85bd0", "#55206e", "фиолетовый шкаф")

    camera = config.camera.position
    mirror_hit = (-0.35, -0.15)
    floor_hit = (0.2, 0.55)
    light = scene.point_lights[0].position
    _arrow(ax, (camera.x, camera.z), mirror_hit, "#e4573d", "1. camera ray")
    _arrow(ax, mirror_hit, floor_hit, "#2d9cdb", "2. mirror bounce")
    _arrow(ax, floor_hit, (light.x, light.z), "#f2994a", "3. shadow ray к point light")
    _arrow(ax, floor_hit, (-1.2, 1.2), "#777777", "4. случайный diffuse bounce")
    ax.scatter([camera.x], [camera.z], s=130, color="#e4573d", edgecolors="white", zorder=8)
    ax.scatter([light.x], [light.z], s=160, color="#f2994a", edgecolors="#401900", zorder=8)
    ax.text(0.03, 0.04, "Если shadow ray перекрыт объектом, прямой вклад света = 0. Если открыт, добавляем вклад с делением на pdf.", transform=ax.transAxes, fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "04_monte_carlo_path.png", dpi=180)
    plt.close(fig)


def draw_material_model(config) -> None:
    materials = {material.name: material for material in config.materials}
    names = ["white", "blue_box", "mirror", "wood", "violet"]
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 7), facecolor="#101720")
    fig.suptitle("Материал: diffuse + mirror <= 1 по каждому RGB каналу", color="white", fontsize=18, weight="bold")
    for ax, name in zip(axes, names):
        material = materials[name]
        diffuse = np.array(material.diffuse.to_tuple())
        mirror = np.array(material.mirror.to_tuple())
        ax.set_facecolor("#101720")
        ax.barh([0.2, 0.6, 1.0], diffuse, height=0.16, color=["#e74c3c", "#2ecc71", "#3498db"], alpha=0.72, label="diffuse")
        ax.barh([0.2, 0.6, 1.0], mirror, left=diffuse, height=0.16, color=["#ffb3aa", "#a7f5c1", "#a8d4ff"], alpha=0.92, label="mirror")
        ax.axvline(1.0, color="#ffffff", alpha=0.5, linestyle="--")
        ax.set_xlim(0.0, 1.05)
        ax.set_yticks([0.2, 0.6, 1.0], ["R", "G", "B"], color="white")
        ax.tick_params(axis="x", colors="white")
        ax.set_title(name, color="white", loc="left", fontsize=12, pad=2)
        ax.grid(axis="x", alpha=0.15)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.text(0.5, 0.02, "Зеленое зеркало сделано не большим diffuse, а цветным mirror: оно отражает зеленый канал сильнее.", ha="center", color="#d7e3f4", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(ASSETS / "05_material_model.png", dpi=180)
    plt.close(fig)


def draw_accumulation_examples(config) -> None:
    renderer = TaichiProgressivePathTracer()
    render = config.render.__class__(
        width=256,
        height=256,
        samples_per_pixel=config.render.samples_per_pixel,
        max_depth=config.render.max_depth,
        min_depth=config.render.min_depth,
        gamma=config.render.gamma,
        normalization=config.render.normalization,
        normalization_value=config.render.normalization_value,
        seed=config.render.seed,
        background=config.render.background,
    )
    small = config.__class__(
        camera=config.camera,
        render=render,
        materials=config.materials,
        triangles=config.triangles,
        obj_meshes=config.obj_meshes,
        point_lights=config.point_lights,
    )
    frames = [1, 4, 16]
    images = []
    for frame_count in frames:
        artifact = renderer.render_frames(small, frames=frame_count, config_path=CONFIG)
        images.append(artifact.display.astype(np.uint8))
        save_png(ASSETS / f"06_accumulation_spp_{frame_count}.png", artifact.display)
    _make_accumulation_panel(frames, images)


def draw_accumulation_concept() -> None:
    rng = np.random.default_rng(42)
    base = np.linspace(0.15, 0.85, 256)[None, :]
    base = np.repeat(base, 256, axis=0)
    images = []
    frames = [1, 4, 16]
    for frame_count in frames:
        noise = rng.normal(0.0, 0.22 / np.sqrt(frame_count), size=(256, 256))
        gray = np.clip(base + noise, 0.0, 1.0)
        images.append((np.dstack([gray * 0.9, gray, gray * 1.05]) * 255).astype(np.uint8))
    _make_accumulation_panel(frames, images)


def _make_accumulation_panel(frames, images) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4), facecolor="#101720")
    fig.suptitle("Накопление SPP: шум падает как 1/sqrt(N)", color="white", fontsize=18, weight="bold")
    for ax, frame_count, image in zip(axes, frames, images):
        ax.imshow(image)
        ax.set_title(f"SPP = {frame_count}", color="white", fontsize=13)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(ASSETS / "06_accumulation_spp.png", dpi=180)
    plt.close(fig)


def _bounds(triangles):
    points = []
    for tri in triangles:
        points.extend([tri.a.to_tuple(), tri.b.to_tuple(), tri.c.to_tuple()])
    arr = np.asarray(points, dtype=float)
    return arr[:, 0].min(), arr[:, 2].min(), arr[:, 0].max(), arr[:, 2].max()


def _center(tri):
    return np.mean(np.asarray([tri.a.to_tuple(), tri.b.to_tuple(), tri.c.to_tuple()], dtype=float), axis=0)


def _rect(ax, bbox, color, edge, label, alpha=0.7):
    x0, z0, x1, z1 = bbox
    patch = Rectangle((x0, z0), x1 - x0, z1 - z0, facecolor=color, edgecolor=edge, linewidth=1.8, alpha=alpha, label=label)
    ax.add_patch(patch)


def _box(ax, x, y, w, h, title, body, color):
    rect = Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.92)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.68, title, transform=ax.transAxes, ha="center", color="white", fontsize=13, weight="bold")
    ax.text(x + w / 2, y + h * 0.34, body, transform=ax.transAxes, ha="center", color="white", fontsize=10)


def _arrow(ax, start, end, color, label):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=17, linewidth=2.3, color=color, label=label))


if __name__ == "__main__":
    main()

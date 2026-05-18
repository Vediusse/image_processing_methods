from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple

import numpy as np

from image_lab4.models.scene import RenderArtifact, SceneConfig
from image_lab4.services.path_tracer import PathTracer


class RealtimePreviewRenderer:
    """Fast raster preview for interactive scene inspection.

    This renderer is intentionally separate from the path tracer: it uses the
    same scene data, materials, and area lights, but estimates lighting with a
    z-buffer raster pass instead of Monte Carlo paths.
    """

    def __init__(self) -> None:
        self._scene_builder = PathTracer()

    def render(self, config: SceneConfig, config_path: Optional[Path] = None, max_size: int = 640) -> RenderArtifact:
        started = perf_counter()
        scene = self._scene_builder._build_scene(config, strict_resolution=False)
        width, height = _preview_size(config.render.width, config.render.height, max_size)
        radiance = self._rasterize(scene, width, height)
        display = self._tone_map(radiance, gamma=scene.render.gamma, percentile=99.0)
        elapsed = max(perf_counter() - started, 1e-9)
        fps = 1.0 / elapsed
        summary = (
            "Realtime preview завершен.\n"
            "Backend: numpy z-buffer raster\n"
            "Треугольников: {0}\n"
            "Источников света: {1}\n"
            "Разрешение: {2}x{3}\n"
            "Время кадра: {4:.2f} ms\n"
            "Оценка FPS: {5:.1f}\n"
            "Важно: это быстрый интерактивный просмотр, финальная физика считается path tracer.".format(
                len(scene.triangles),
                len(scene.lights),
                width,
                height,
                elapsed * 1000.0,
                fps,
            )
        )
        return RenderArtifact(radiance=radiance, display=display, summary=summary, scene=scene, config_path=config_path)

    def _rasterize(self, scene, width: int, height: int) -> np.ndarray:
        aspect = width / height
        camera = np.asarray(scene.camera.position.to_tuple(), dtype=np.float32)
        forward = np.asarray(scene.camera_forward.to_tuple(), dtype=np.float32)
        right = np.asarray(scene.camera_right.to_tuple(), dtype=np.float32)
        up = np.asarray(scene.camera_up.to_tuple(), dtype=np.float32)
        tan_half_fov = float(scene.tan_half_fov)

        radiance = np.zeros((height, width, 3), dtype=np.float32)
        depth = np.full((height, width), np.inf, dtype=np.float32)

        light_data = []
        for light in scene.lights:
            center = _triangle_center(light)
            light_data.append(
                (
                    center,
                    np.asarray(light.normal.to_tuple(), dtype=np.float32),
                    np.asarray(light.emission.to_tuple(), dtype=np.float32),
                    float(light.area),
                )
            )

        for triangle in scene.triangles:
            world = np.asarray([triangle.a.to_tuple(), triangle.b.to_tuple(), triangle.c.to_tuple()], dtype=np.float32)
            projected = _project_triangle(world, camera, forward, right, up, tan_half_fov, aspect, width, height)
            if projected is None:
                continue

            screen, camera_z = projected
            min_x = max(int(np.floor(screen[:, 0].min())), 0)
            max_x = min(int(np.ceil(screen[:, 0].max())), width - 1)
            min_y = max(int(np.floor(screen[:, 1].min())), 0)
            max_y = min(int(np.ceil(screen[:, 1].max())), height - 1)
            if min_x > max_x or min_y > max_y:
                continue

            xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
            ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
            grid_x, grid_y = np.meshgrid(xs, ys)
            bary = _barycentric_grid(grid_x, grid_y, screen)
            mask = np.all(bary >= -1e-5, axis=2)
            if not np.any(mask):
                continue

            local_depth = bary[:, :, 0] * camera_z[0] + bary[:, :, 1] * camera_z[1] + bary[:, :, 2] * camera_z[2]
            depth_view = depth[min_y : max_y + 1, min_x : max_x + 1]
            visible = mask & (local_depth > 1e-5) & (local_depth < depth_view)
            if not np.any(visible):
                continue

            positions = (
                bary[:, :, 0:1] * world[0]
                + bary[:, :, 1:2] * world[1]
                + bary[:, :, 2:3] * world[2]
            )
            normal = np.asarray(triangle.normal.to_tuple(), dtype=np.float32)
            view_direction = camera - positions
            oriented_normal = np.where(np.sum(normal * view_direction, axis=2, keepdims=True) >= 0.0, normal, -normal)
            color = _shade_triangle(triangle, positions, oriented_normal, light_data)

            tile = radiance[min_y : max_y + 1, min_x : max_x + 1]
            tile[visible] = color[visible]
            depth_view[visible] = local_depth[visible]

        return radiance

    def _tone_map(self, radiance: np.ndarray, gamma: float, percentile: float) -> np.ndarray:
        positive = np.clip(radiance, 0.0, None)
        scale = float(np.percentile(positive, percentile))
        scale = max(scale, 1e-8)
        mapped = np.clip(positive / scale, 0.0, 1.0)
        return np.clip(np.power(mapped, 1.0 / gamma) * 255.0, 0.0, 255.0)


def _preview_size(width: int, height: int, max_size: int) -> Tuple[int, int]:
    scale = min(1.0, max_size / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _project_triangle(world: np.ndarray, camera: np.ndarray, forward: np.ndarray, right: np.ndarray, up: np.ndarray, tan_half_fov: float, aspect: float, width: int, height: int):
    relative = world - camera[None, :]
    camera_x = relative @ right
    camera_y = relative @ up
    camera_z = relative @ forward
    if np.any(camera_z <= 1e-4):
        return None
    ndc_x = camera_x / (camera_z * tan_half_fov * aspect)
    ndc_y = camera_y / (camera_z * tan_half_fov)
    screen_x = (ndc_x + 1.0) * 0.5 * width
    screen_y = (1.0 - ndc_y) * 0.5 * height
    return np.stack((screen_x, screen_y), axis=1), camera_z


def _barycentric_grid(grid_x: np.ndarray, grid_y: np.ndarray, screen: np.ndarray) -> np.ndarray:
    ax, ay = screen[0]
    bx, by = screen[1]
    cx, cy = screen[2]
    denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(float(denominator)) < 1e-8:
        return np.full(grid_x.shape + (3,), -1.0, dtype=np.float32)
    w0 = ((by - cy) * (grid_x - cx) + (cx - bx) * (grid_y - cy)) / denominator
    w1 = ((cy - ay) * (grid_x - cx) + (ax - cx) * (grid_y - cy)) / denominator
    w2 = 1.0 - w0 - w1
    return np.stack((w0, w1, w2), axis=2)


def _shade_triangle(triangle, positions: np.ndarray, normals: np.ndarray, light_data) -> np.ndarray:
    emission = np.asarray(triangle.emission.to_tuple(), dtype=np.float32)
    if float(emission.max()) > 0.0:
        return np.broadcast_to(emission, positions.shape).copy()

    diffuse = np.asarray(triangle.material.diffuse.to_tuple(), dtype=np.float32)
    mirror = np.asarray(triangle.material.mirror.to_tuple(), dtype=np.float32)
    color = diffuse * 0.035
    bsdf = diffuse / np.pi
    for light_center, light_normal, light_emission, light_area in light_data:
        to_light = light_center[None, None, :] - positions
        distance_squared = np.maximum(np.sum(to_light * to_light, axis=2, keepdims=True), 1e-6)
        direction = to_light / np.sqrt(distance_squared)
        cosine_surface = np.maximum(np.sum(normals * direction, axis=2, keepdims=True), 0.0)
        cosine_light = np.maximum(np.sum(light_normal[None, None, :] * -direction, axis=2, keepdims=True), 0.0)
        color = color + light_emission[None, None, :] * bsdf[None, None, :] * (
            cosine_surface * cosine_light * light_area / distance_squared
        )

    if float(mirror.max()) > 0.0:
        color = color + mirror[None, None, :] * 0.06
    return color


def _triangle_center(triangle) -> np.ndarray:
    return np.asarray(
        (
            (triangle.a.x + triangle.b.x + triangle.c.x) / 3.0,
            (triangle.a.y + triangle.b.y + triangle.c.y) / 3.0,
            (triangle.a.z + triangle.b.z + triangle.c.z) / 3.0,
        ),
        dtype=np.float32,
    )

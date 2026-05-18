from pathlib import Path
from time import perf_counter
from typing import Dict, Optional

import numpy as np
import taichi as ti

from image_lab4.models.scene import RenderArtifact, SceneConfig
from image_lab4.services.path_tracer import PathTracer


_TAICHI_READY = False


def _ensure_taichi() -> None:
    global _TAICHI_READY
    if _TAICHI_READY:
        return
    ti.init(arch=ti.metal, offline_cache=True, log_level=ti.ERROR)
    _TAICHI_READY = True


class TaichiRealtimeRenderer:
    def __init__(self) -> None:
        _ensure_taichi()
        self._scene_builder = PathTracer()

    def render(self, config: SceneConfig, config_path: Optional[Path] = None) -> RenderArtifact:
        started = perf_counter()
        scene = self._scene_builder._build_scene(config, strict_resolution=False)
        arrays = _scene_arrays(scene)
        width = int(config.render.width)
        height = int(config.render.height)
        radiance = np.zeros((height, width, 3), dtype=np.float32)
        display = np.zeros((height, width, 3), dtype=np.float32)

        _render_realtime_kernel(
            width,
            height,
            len(scene.triangles),
            len(scene.lights),
            float(scene.tan_half_fov),
            float(width) / float(height),
            float(config.render.gamma),
            np.asarray(scene.camera.position.to_tuple(), dtype=np.float32),
            np.asarray(scene.camera_forward.to_tuple(), dtype=np.float32),
            np.asarray(scene.camera_right.to_tuple(), dtype=np.float32),
            np.asarray(scene.camera_up.to_tuple(), dtype=np.float32),
            arrays["a"],
            arrays["edge1"],
            arrays["edge2"],
            arrays["normals"],
            arrays["diffuse"],
            arrays["mirror"],
            arrays["emission"],
            arrays["light_centers"],
            arrays["light_normals"],
            arrays["light_emission"],
            arrays["light_areas"],
            radiance,
            display,
        )

        elapsed = max(perf_counter() - started, 1e-9)
        summary = (
            "Taichi realtime preview завершен.\n"
            "Backend: Taichi Metal GPGPU\n"
            "Треугольников: {0}\n"
            "Источников света: {1}\n"
            "Разрешение: {2}x{3}\n"
            "Время кадра: {4:.2f} ms\n"
            "Оценка FPS: {5:.1f}\n"
            "Важно: это realtime GPU-preview; финальный path tracing/HDR остается отдельным режимом.".format(
                len(scene.triangles),
                len(scene.lights),
                width,
                height,
                elapsed * 1000.0,
                1.0 / elapsed,
            )
        )
        return RenderArtifact(radiance=radiance, display=display, summary=summary, scene=scene, config_path=config_path)


def _scene_arrays(scene) -> Dict[str, np.ndarray]:
    a = np.asarray(scene.triangle_a, dtype=np.float32)
    edge1 = np.asarray(scene.triangle_edge1, dtype=np.float32)
    edge2 = np.asarray(scene.triangle_edge2, dtype=np.float32)
    normals = np.asarray(scene.triangle_normals, dtype=np.float32)
    diffuse = np.asarray([item.material.diffuse.to_tuple() for item in scene.triangles], dtype=np.float32)
    mirror = np.asarray([item.material.mirror.to_tuple() for item in scene.triangles], dtype=np.float32)
    emission = np.asarray([item.emission.to_tuple() for item in scene.triangles], dtype=np.float32)
    light_centers = []
    light_normals = []
    light_emission = []
    light_areas = []
    for light in scene.lights:
        light_centers.append(
            (
                (light.a.x + light.b.x + light.c.x) / 3.0,
                (light.a.y + light.b.y + light.c.y) / 3.0,
                (light.a.z + light.b.z + light.c.z) / 3.0,
            )
        )
        light_normals.append(light.normal.to_tuple())
        light_emission.append(light.emission.to_tuple())
        light_areas.append(light.area)
    return {
        "a": a,
        "edge1": edge1,
        "edge2": edge2,
        "normals": normals,
        "diffuse": diffuse,
        "mirror": mirror,
        "emission": emission,
        "light_centers": np.asarray(light_centers, dtype=np.float32),
        "light_normals": np.asarray(light_normals, dtype=np.float32),
        "light_emission": np.asarray(light_emission, dtype=np.float32),
        "light_areas": np.asarray(light_areas, dtype=np.float32),
    }


@ti.func
def _dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


@ti.func
def _cross(a, b):
    return ti.Vector(
        (
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        )
    )


@ti.func
def _normalize(v):
    return v / ti.max(ti.sqrt(_dot(v, v)), 1e-8)


@ti.kernel
def _render_realtime_kernel(
    width: ti.i32,
    height: ti.i32,
    triangle_count: ti.i32,
    light_count: ti.i32,
    tan_half_fov: ti.f32,
    aspect: ti.f32,
    gamma: ti.f32,
    camera: ti.types.ndarray(dtype=ti.f32, ndim=1),
    forward: ti.types.ndarray(dtype=ti.f32, ndim=1),
    right: ti.types.ndarray(dtype=ti.f32, ndim=1),
    up: ti.types.ndarray(dtype=ti.f32, ndim=1),
    tri_a: ti.types.ndarray(dtype=ti.f32, ndim=2),
    tri_edge1: ti.types.ndarray(dtype=ti.f32, ndim=2),
    tri_edge2: ti.types.ndarray(dtype=ti.f32, ndim=2),
    tri_normals: ti.types.ndarray(dtype=ti.f32, ndim=2),
    mat_diffuse: ti.types.ndarray(dtype=ti.f32, ndim=2),
    mat_mirror: ti.types.ndarray(dtype=ti.f32, ndim=2),
    tri_emission: ti.types.ndarray(dtype=ti.f32, ndim=2),
    light_centers: ti.types.ndarray(dtype=ti.f32, ndim=2),
    light_normals: ti.types.ndarray(dtype=ti.f32, ndim=2),
    light_emission: ti.types.ndarray(dtype=ti.f32, ndim=2),
    light_areas: ti.types.ndarray(dtype=ti.f32, ndim=1),
    radiance: ti.types.ndarray(dtype=ti.f32, ndim=3),
    display: ti.types.ndarray(dtype=ti.f32, ndim=3),
):
    camera_v = ti.Vector((camera[0], camera[1], camera[2]))
    forward_v = ti.Vector((forward[0], forward[1], forward[2]))
    right_v = ti.Vector((right[0], right[1], right[2]))
    up_v = ti.Vector((up[0], up[1], up[2]))

    for y, x in ti.ndrange(height, width):
        sample_x = (ti.cast(x, ti.f32) + 0.5) / ti.cast(width, ti.f32)
        sample_y = (ti.cast(y, ti.f32) + 0.5) / ti.cast(height, ti.f32)
        ndc_x = (2.0 * sample_x - 1.0) * aspect * tan_half_fov
        ndc_y = (1.0 - 2.0 * sample_y) * tan_half_fov
        ray_dir = _normalize(forward_v + right_v * ndc_x + up_v * ndc_y)

        closest_t = 1.0e20
        hit_index = -1
        hit_u = 0.0
        hit_v = 0.0

        for i in range(triangle_count):
            a = ti.Vector((tri_a[i, 0], tri_a[i, 1], tri_a[i, 2]))
            edge1 = ti.Vector((tri_edge1[i, 0], tri_edge1[i, 1], tri_edge1[i, 2]))
            edge2 = ti.Vector((tri_edge2[i, 0], tri_edge2[i, 1], tri_edge2[i, 2]))
            pvec = _cross(ray_dir, edge2)
            det = _dot(edge1, pvec)
            if ti.abs(det) > 1e-8:
                inv_det = 1.0 / det
                tvec = camera_v - a
                u = _dot(tvec, pvec) * inv_det
                if u >= 0.0 and u <= 1.0:
                    qvec = _cross(tvec, edge1)
                    v = _dot(ray_dir, qvec) * inv_det
                    if v >= 0.0 and u + v <= 1.0:
                        t = _dot(edge2, qvec) * inv_det
                        if t > 1e-5 and t < closest_t:
                            closest_t = t
                            hit_index = i
                            hit_u = u
                            hit_v = v

        color = ti.Vector((0.0, 0.0, 0.0))
        if hit_index >= 0:
            hit_pos = camera_v + ray_dir * closest_t
            normal = ti.Vector((tri_normals[hit_index, 0], tri_normals[hit_index, 1], tri_normals[hit_index, 2]))
            if _dot(normal, -ray_dir) < 0.0:
                normal = -normal

            emission = ti.Vector((tri_emission[hit_index, 0], tri_emission[hit_index, 1], tri_emission[hit_index, 2]))
            diffuse = ti.Vector((mat_diffuse[hit_index, 0], mat_diffuse[hit_index, 1], mat_diffuse[hit_index, 2]))
            mirror = ti.Vector((mat_mirror[hit_index, 0], mat_mirror[hit_index, 1], mat_mirror[hit_index, 2]))
            if ti.max(emission.x, emission.y, emission.z) > 0.0:
                color = emission
            else:
                color = diffuse * 0.035
                bsdf = diffuse / 3.14159265
                for l in range(light_count):
                    light_center = ti.Vector((light_centers[l, 0], light_centers[l, 1], light_centers[l, 2]))
                    light_normal = ti.Vector((light_normals[l, 0], light_normals[l, 1], light_normals[l, 2]))
                    light_power = ti.Vector((light_emission[l, 0], light_emission[l, 1], light_emission[l, 2]))
                    to_light = light_center - hit_pos
                    dist2 = ti.max(_dot(to_light, to_light), 1e-6)
                    light_dir = to_light / ti.sqrt(dist2)
                    cosine_surface = ti.max(_dot(normal, light_dir), 0.0)
                    cosine_light = ti.max(_dot(light_normal, -light_dir), 0.0)
                    color += light_power * bsdf * (cosine_surface * cosine_light * light_areas[l] / dist2)
                color += mirror * 0.045 * ti.max(_dot(normal, -ray_dir), 0.0)

        # Fixed exposure keeps the preview stable frame-to-frame and avoids a CPU percentile pass.
        exposure = 12.0
        mapped = ti.min(color * exposure, ti.Vector((1.0, 1.0, 1.0)))
        mapped = ti.pow(ti.max(mapped, ti.Vector((0.0, 0.0, 0.0))), ti.Vector((1.0 / gamma, 1.0 / gamma, 1.0 / gamma)))

        radiance[y, x, 0] = color.x
        radiance[y, x, 1] = color.y
        radiance[y, x, 2] = color.z
        display[y, x, 0] = mapped.x * 255.0
        display[y, x, 1] = mapped.y * 255.0
        display[y, x, 2] = mapped.z * 255.0

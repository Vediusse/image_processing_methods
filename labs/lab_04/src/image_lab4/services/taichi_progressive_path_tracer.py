from pathlib import Path
from time import perf_counter
from typing import Dict, Optional

import numpy as np
import taichi as ti

from image_lab4.models.scene import RenderArtifact, SceneConfig
from image_lab4.services.image_filters import FilterGuide, FilterSettings, split_bilateral_denoise
from image_lab4.services.path_tracer import PathTracer
from image_lab4.services.taichi_realtime import _ensure_taichi


class TaichiProgressivePathTracer:
    def __init__(self) -> None:
        _ensure_taichi()
        self._scene_builder = PathTracer()

    def create_state(self, config: SceneConfig, config_path: Optional[Path] = None) -> Dict[str, object]:
        scene = self._scene_builder._build_scene(config, strict_resolution=False)
        width = int(config.render.width)
        height = int(config.render.height)
        return {
            "config": config,
            "config_path": config_path,
            "scene": scene,
            "arrays": _scene_arrays(scene),
            "frame_radiance": np.zeros((height, width, 3), dtype=np.float32),
            "frame_direct": np.zeros((height, width, 3), dtype=np.float32),
            "width": width,
            "height": height,
        }

    def trace_state_frame(self, state: Dict[str, object], seed_offset: int) -> np.ndarray:
        config = state["config"]
        scene = state["scene"]
        arrays = state["arrays"]
        frame_radiance = state["frame_radiance"]
        frame_direct = state["frame_direct"]
        width = int(state["width"])
        height = int(state["height"])
        _path_trace_frame_kernel(
            width,
            height,
            len(scene.triangles),
            len(scene.lights),
            len(scene.point_lights),
            int(config.render.max_depth),
            int(config.render.min_depth),
            int(config.render.seed + seed_offset * 9781),
            float(scene.tan_half_fov),
            float(width) / float(height),
            np.asarray(scene.camera.position.to_tuple(), dtype=np.float32),
            np.asarray(scene.camera_forward.to_tuple(), dtype=np.float32),
            np.asarray(scene.camera_right.to_tuple(), dtype=np.float32),
            np.asarray(scene.camera_up.to_tuple(), dtype=np.float32),
            np.asarray(config.render.background.to_tuple(), dtype=np.float32),
            arrays["a"],
            arrays["edge1"],
            arrays["edge2"],
            arrays["normals"],
            arrays["diffuse"],
            arrays["mirror"],
            arrays["emission"],
            arrays["light_a"],
            arrays["light_edge1"],
            arrays["light_edge2"],
            arrays["light_normals"],
            arrays["light_emission"],
            arrays["light_areas"],
            arrays["point_positions"],
            arrays["point_intensities"],
            arrays["source_cdf"],
            frame_radiance,
            frame_direct,
        )
        return frame_radiance

    def render_frames(
        self,
        config: SceneConfig,
        frames: int,
        config_path: Optional[Path] = None,
        seed_offset: int = 0,
    ) -> RenderArtifact:
        started = perf_counter()
        scene = self._scene_builder._build_scene(config, strict_resolution=False)
        arrays = _scene_arrays(scene)
        width = int(config.render.width)
        height = int(config.render.height)
        frame_radiance = np.zeros((height, width, 3), dtype=np.float32)
        frame_direct = np.zeros((height, width, 3), dtype=np.float32)
        accumulation = np.zeros((height, width, 3), dtype=np.float32)
        direct_accumulation = np.zeros((height, width, 3), dtype=np.float32)
        display = np.zeros((height, width, 3), dtype=np.float32)
        frame_count = max(1, int(frames))
        times = []

        for frame_index in range(frame_count):
            frame_started = perf_counter()
            _path_trace_frame_kernel(
                width,
                height,
                len(scene.triangles),
                len(scene.lights),
                len(scene.point_lights),
                int(config.render.max_depth),
                int(config.render.min_depth),
                int(config.render.seed + (seed_offset + frame_index) * 9781),
                float(scene.tan_half_fov),
                float(width) / float(height),
                np.asarray(scene.camera.position.to_tuple(), dtype=np.float32),
                np.asarray(scene.camera_forward.to_tuple(), dtype=np.float32),
                np.asarray(scene.camera_right.to_tuple(), dtype=np.float32),
                np.asarray(scene.camera_up.to_tuple(), dtype=np.float32),
                np.asarray(config.render.background.to_tuple(), dtype=np.float32),
                arrays["a"],
                arrays["edge1"],
                arrays["edge2"],
                arrays["normals"],
                arrays["diffuse"],
                arrays["mirror"],
                arrays["emission"],
                arrays["light_a"],
                arrays["light_edge1"],
                arrays["light_edge2"],
                arrays["light_normals"],
                arrays["light_emission"],
                arrays["light_areas"],
                arrays["point_positions"],
                arrays["point_intensities"],
                arrays["source_cdf"],
                frame_radiance,
                frame_direct,
            )
            accumulation += frame_radiance
            direct_accumulation += frame_direct
            times.append(perf_counter() - frame_started)

        radiance = accumulation / float(frame_count)
        direct = direct_accumulation / float(frame_count)
        denoise_summary = "Фильтр: выключен"
        if config.denoise.enabled and config.denoise.filter_name.lower() != "none":
            depth, normals, object_ids = _build_gbuffer(scene, arrays, width, height)
            secondary = np.clip(radiance - direct, 0.0, None)
            denoise_settings = FilterSettings(
                name=config.denoise.filter_name,
                radius=config.denoise.radius,
                sigma_spatial=config.denoise.sigma_spatial,
                sigma_color=config.denoise.sigma_color,
                sigma_depth=config.denoise.sigma_depth,
                sigma_normal=config.denoise.sigma_normal,
                strength=config.denoise.strength,
                preserve_object_flux=config.denoise.preserve_object_flux,
            )
            radiance = split_bilateral_denoise(direct, secondary, denoise_settings, FilterGuide(depth=depth, normals=normals, object_ids=object_ids))
            denoise_summary = (
                "Фильтр: {0}, radius={1}, strength={2:.2f}, guide=depth/object/normal".format(
                    config.denoise.filter_name,
                    config.denoise.radius,
                    config.denoise.strength,
                )
            )
        display[:] = _tone_map(radiance, config.render.gamma, config.render.normalization, config.render.normalization_value)
        elapsed = max(perf_counter() - started, 1e-9)
        warm_times = times[1:] or times
        warm_average = sum(warm_times) / len(warm_times)
        summary = (
            "Taichi progressive path tracing завершен.\n"
            "Backend: Taichi Metal Monte-Carlo path tracing\n"
            "Треугольников: {0}\n"
            "Area источников: {1}\n"
            "Point источников: {2}\n"
            "Разрешение: {3}x{4}\n"
            "Frames/SPP накоплено: {5}\n"
            "Max depth: {6}\n"
            "{7}\n"
            "Общее время: {8:.2f} ms\n"
            "Warm frame average: {9:.2f} ms\n"
            "Warm FPS: {10:.1f}".format(
                len(scene.triangles),
                len(scene.lights),
                len(scene.point_lights),
                width,
                height,
                frame_count,
                config.render.max_depth,
                denoise_summary,
                elapsed * 1000.0,
                warm_average * 1000.0,
                1.0 / max(warm_average, 1e-9),
            )
        )
        return RenderArtifact(radiance=radiance, display=display, summary=summary, scene=scene, config_path=config_path)


def _scene_arrays(scene) -> Dict[str, np.ndarray]:
    light_a = []
    light_edge1 = []
    light_edge2 = []
    light_normals = []
    light_emission = []
    light_areas = []
    for light in scene.lights:
        a = np.asarray(light.a.to_tuple(), dtype=np.float32)
        b = np.asarray(light.b.to_tuple(), dtype=np.float32)
        c = np.asarray(light.c.to_tuple(), dtype=np.float32)
        light_a.append(a)
        light_edge1.append(b - a)
        light_edge2.append(c - a)
        light_normals.append(light.normal.to_tuple())
        light_emission.append(light.emission.to_tuple())
        light_areas.append(light.area)
    point_positions = np.asarray([item.position.to_tuple() for item in scene.point_lights], dtype=np.float32).reshape(-1, 3)
    point_intensities = np.asarray([item.intensity.to_tuple() for item in scene.point_lights], dtype=np.float32).reshape(-1, 3)
    source_cdf = np.cumsum(scene.light_probabilities).astype(np.float32)
    if source_cdf.size:
        source_cdf[-1] = 1.0
    return {
        "a": np.asarray(scene.triangle_a, dtype=np.float32),
        "edge1": np.asarray(scene.triangle_edge1, dtype=np.float32),
        "edge2": np.asarray(scene.triangle_edge2, dtype=np.float32),
        "normals": np.asarray(scene.triangle_normals, dtype=np.float32),
        "diffuse": np.asarray([item.material.diffuse.to_tuple() for item in scene.triangles], dtype=np.float32),
        "mirror": np.asarray([item.material.mirror.to_tuple() for item in scene.triangles], dtype=np.float32),
        "emission": np.asarray([item.emission.to_tuple() for item in scene.triangles], dtype=np.float32),
        "light_a": np.asarray(light_a, dtype=np.float32).reshape(-1, 3),
        "light_edge1": np.asarray(light_edge1, dtype=np.float32).reshape(-1, 3),
        "light_edge2": np.asarray(light_edge2, dtype=np.float32).reshape(-1, 3),
        "light_normals": np.asarray(light_normals, dtype=np.float32).reshape(-1, 3),
        "light_emission": np.asarray(light_emission, dtype=np.float32).reshape(-1, 3),
        "light_areas": np.asarray(light_areas, dtype=np.float32).reshape(-1),
        "point_positions": point_positions,
        "point_intensities": point_intensities,
        "source_cdf": source_cdf,
    }


def _tone_map(radiance: np.ndarray, gamma: float, mode: str, value: float) -> np.ndarray:
    positive = np.clip(radiance, 0.0, None)
    if mode == "percentile":
        scale = float(np.percentile(positive, min(max(value, 50.0), 99.9)))
    elif mode == "mean":
        scale = float(np.mean(positive)) / max(value, 1e-8)
    else:
        scale = float(np.max(positive))
    mapped = np.clip(positive / max(scale, 1e-8), 0.0, 1.0)
    return np.clip(np.power(mapped, 1.0 / gamma) * 255.0, 0.0, 255.0)


def _build_gbuffer(scene, arrays: Dict[str, np.ndarray], width: int, height: int):
    depth = np.full((height, width), np.inf, dtype=np.float32)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    object_ids = np.full((height, width), -1, dtype=np.int32)
    _first_hit_gbuffer_kernel(
        width,
        height,
        len(scene.triangles),
        float(scene.tan_half_fov),
        float(width) / float(height),
        np.asarray(scene.camera.position.to_tuple(), dtype=np.float32),
        np.asarray(scene.camera_forward.to_tuple(), dtype=np.float32),
        np.asarray(scene.camera_right.to_tuple(), dtype=np.float32),
        np.asarray(scene.camera_up.to_tuple(), dtype=np.float32),
        arrays["a"],
        arrays["edge1"],
        arrays["edge2"],
        arrays["normals"],
        depth,
        normals,
        object_ids,
    )
    return depth, normals, object_ids


@ti.func
def _hash_u32(x):
    x = (x ^ 61) ^ (x >> 16)
    x = x + (x << 3)
    x = x ^ (x >> 4)
    x = x * 668265263
    x = x ^ (x >> 15)
    return x


@ti.func
def _rand(seed: ti.i32, index: ti.i32, bounce: ti.i32, channel: ti.i32):
    h = _hash_u32(seed + index * 747796405 + bounce * 1103515245 + channel * 277803737)
    return ti.cast(h & 0x00FFFFFF, ti.f32) / 16777216.0


@ti.func
def _dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


@ti.func
def _cross(a, b):
    return ti.Vector((a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x))


@ti.func
def _normalize(v):
    return v / ti.max(ti.sqrt(_dot(v, v)), 1e-8)


@ti.func
def _read3(arr: ti.template(), i):
    return ti.Vector((arr[i, 0], arr[i, 1], arr[i, 2]))


@ti.func
def _cosine_direction(normal, u1, u2):
    r = ti.sqrt(ti.max(u1, 0.0))
    phi = 6.28318530718 * u2
    x = r * ti.cos(phi)
    y = r * ti.sin(phi)
    z = ti.sqrt(ti.max(1.0 - u1, 0.0))
    tangent = ti.Vector((0.0, 0.0, 0.0))
    if ti.abs(normal.x) > 0.1:
        tangent = _normalize(_cross(ti.Vector((0.0, 1.0, 0.0)), normal))
    else:
        tangent = _normalize(_cross(ti.Vector((1.0, 0.0, 0.0)), normal))
    bitangent = _normalize(_cross(normal, tangent))
    return _normalize(tangent * x + bitangent * y + normal * z)


@ti.func
def _intersect(origin, direction, triangle_count, tri_a: ti.template(), tri_edge1: ti.template(), tri_edge2: ti.template()):
    closest_t = 1.0e20
    hit_index = -1
    for i in range(triangle_count):
        a = _read3(tri_a, i)
        edge1 = _read3(tri_edge1, i)
        edge2 = _read3(tri_edge2, i)
        pvec = _cross(direction, edge2)
        det = _dot(edge1, pvec)
        if ti.abs(det) > 1e-8:
            inv_det = 1.0 / det
            tvec = origin - a
            u = _dot(tvec, pvec) * inv_det
            if u >= 0.0 and u <= 1.0:
                qvec = _cross(tvec, edge1)
                v = _dot(direction, qvec) * inv_det
                if v >= 0.0 and u + v <= 1.0:
                    t = _dot(edge2, qvec) * inv_det
                    if t > 1e-4 and t < closest_t:
                        closest_t = t
                        hit_index = i
    return hit_index, closest_t


@ti.kernel
def _first_hit_gbuffer_kernel(
    width: ti.i32,
    height: ti.i32,
    triangle_count: ti.i32,
    tan_half_fov: ti.f32,
    aspect: ti.f32,
    camera: ti.types.ndarray(dtype=ti.f32, ndim=1),
    forward: ti.types.ndarray(dtype=ti.f32, ndim=1),
    right: ti.types.ndarray(dtype=ti.f32, ndim=1),
    up: ti.types.ndarray(dtype=ti.f32, ndim=1),
    tri_a: ti.types.ndarray(dtype=ti.f32, ndim=2),
    tri_edge1: ti.types.ndarray(dtype=ti.f32, ndim=2),
    tri_edge2: ti.types.ndarray(dtype=ti.f32, ndim=2),
    tri_normals: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out_depth: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out_normals: ti.types.ndarray(dtype=ti.f32, ndim=3),
    out_object_ids: ti.types.ndarray(dtype=ti.i32, ndim=2),
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
        direction = _normalize(forward_v + right_v * ndc_x + up_v * ndc_y)
        hit_index, hit_t = _intersect(camera_v, direction, triangle_count, tri_a, tri_edge1, tri_edge2)
        out_depth[y, x] = hit_t
        out_object_ids[y, x] = hit_index
        if hit_index >= 0:
            normal = _read3(tri_normals, hit_index)
            if _dot(normal, direction) > 0.0:
                normal = -normal
            out_normals[y, x, 0] = normal.x
            out_normals[y, x, 1] = normal.y
            out_normals[y, x, 2] = normal.z
        else:
            out_object_ids[y, x] = -1
            out_normals[y, x, 0] = 0.0
            out_normals[y, x, 1] = 0.0
            out_normals[y, x, 2] = 0.0


@ti.kernel
def _path_trace_frame_kernel(
    width: ti.i32,
    height: ti.i32,
    triangle_count: ti.i32,
    light_count: ti.i32,
    point_light_count: ti.i32,
    max_depth: ti.i32,
    min_depth: ti.i32,
    seed: ti.i32,
    tan_half_fov: ti.f32,
    aspect: ti.f32,
    camera: ti.types.ndarray(dtype=ti.f32, ndim=1),
    forward: ti.types.ndarray(dtype=ti.f32, ndim=1),
    right: ti.types.ndarray(dtype=ti.f32, ndim=1),
    up: ti.types.ndarray(dtype=ti.f32, ndim=1),
    background: ti.types.ndarray(dtype=ti.f32, ndim=1),
    tri_a: ti.types.ndarray(dtype=ti.f32, ndim=2),
    tri_edge1: ti.types.ndarray(dtype=ti.f32, ndim=2),
    tri_edge2: ti.types.ndarray(dtype=ti.f32, ndim=2),
    tri_normals: ti.types.ndarray(dtype=ti.f32, ndim=2),
    mat_diffuse: ti.types.ndarray(dtype=ti.f32, ndim=2),
    mat_mirror: ti.types.ndarray(dtype=ti.f32, ndim=2),
    tri_emission: ti.types.ndarray(dtype=ti.f32, ndim=2),
    light_a: ti.types.ndarray(dtype=ti.f32, ndim=2),
    light_edge1: ti.types.ndarray(dtype=ti.f32, ndim=2),
    light_edge2: ti.types.ndarray(dtype=ti.f32, ndim=2),
    light_normals: ti.types.ndarray(dtype=ti.f32, ndim=2),
    light_emission: ti.types.ndarray(dtype=ti.f32, ndim=2),
    light_areas: ti.types.ndarray(dtype=ti.f32, ndim=1),
    point_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),
    point_intensities: ti.types.ndarray(dtype=ti.f32, ndim=2),
    source_cdf: ti.types.ndarray(dtype=ti.f32, ndim=1),
    out_radiance: ti.types.ndarray(dtype=ti.f32, ndim=3),
    out_direct: ti.types.ndarray(dtype=ti.f32, ndim=3),
):
    camera_v = ti.Vector((camera[0], camera[1], camera[2]))
    forward_v = ti.Vector((forward[0], forward[1], forward[2]))
    right_v = ti.Vector((right[0], right[1], right[2]))
    up_v = ti.Vector((up[0], up[1], up[2]))
    background_v = ti.Vector((background[0], background[1], background[2]))

    for y, x in ti.ndrange(height, width):
        pixel_index = y * width + x
        jitter_x = _rand(seed, pixel_index, 0, 1)
        jitter_y = _rand(seed, pixel_index, 0, 2)
        sample_x = (ti.cast(x, ti.f32) + jitter_x) / ti.cast(width, ti.f32)
        sample_y = (ti.cast(y, ti.f32) + jitter_y) / ti.cast(height, ti.f32)
        ndc_x = (2.0 * sample_x - 1.0) * aspect * tan_half_fov
        ndc_y = (1.0 - 2.0 * sample_y) * tan_half_fov
        origin = camera_v
        direction = _normalize(forward_v + right_v * ndc_x + up_v * ndc_y)
        throughput = ti.Vector((1.0, 1.0, 1.0))
        radiance = ti.Vector((0.0, 0.0, 0.0))
        first_bounce_direct = ti.Vector((0.0, 0.0, 0.0))
        active = True
        last_specular = True

        for depth in range(max_depth):
            if active:
                hit_index, hit_t = _intersect(origin, direction, triangle_count, tri_a, tri_edge1, tri_edge2)
                if hit_index < 0:
                    radiance += throughput * background_v
                    active = False
                else:
                    hit_pos = origin + direction * hit_t
                    normal = _read3(tri_normals, hit_index)
                    if _dot(normal, direction) > 0.0:
                        normal = -normal
                    emission = _read3(tri_emission, hit_index)
                    diffuse = _read3(mat_diffuse, hit_index)
                    mirror = _read3(mat_mirror, hit_index)
                    if ti.max(emission.x, emission.y, emission.z) > 0.0 and last_specular:
                        radiance += throughput * emission

                    source_count = light_count + point_light_count
                    if ti.max(diffuse.x, diffuse.y, diffuse.z) > 0.0 and source_count > 0:
                        pick = _rand(seed, pixel_index, depth, 3)
                        slot = source_count - 1
                        for source_index in range(source_count):
                            if pick <= source_cdf[source_index] and slot == source_count - 1:
                                slot = source_index
                        source_probability = source_cdf[slot]
                        if slot > 0:
                            source_probability = source_cdf[slot] - source_cdf[slot - 1]
                        bsdf = diffuse / 3.14159265

                        if slot < light_count:
                            la = _read3(light_a, slot)
                            le1 = _read3(light_edge1, slot)
                            le2 = _read3(light_edge2, slot)
                            lu1 = _rand(seed, pixel_index, depth, 4)
                            lu2 = _rand(seed, pixel_index, depth, 5)
                            slu1 = ti.sqrt(lu1)
                            alpha = 1.0 - slu1
                            beta = slu1 * (1.0 - lu2)
                            gamma = slu1 * lu2
                            light_point = la * alpha + (la + le1) * beta + (la + le2) * gamma
                            to_light = light_point - hit_pos
                            dist2 = ti.max(_dot(to_light, to_light), 1e-6)
                            light_dir = to_light / ti.sqrt(dist2)
                            cos_surface = ti.max(_dot(normal, light_dir), 0.0)
                            light_normal = _read3(light_normals, slot)
                            cos_light = ti.max(_dot(light_normal, -light_dir), 0.0)
                            if cos_surface > 0.0 and cos_light > 0.0:
                                shadow_origin = hit_pos + light_dir * 1e-4
                                shadow_index, shadow_t = _intersect(shadow_origin, light_dir, triangle_count, tri_a, tri_edge1, tri_edge2)
                                target_distance = ti.sqrt(dist2)
                                if shadow_index >= 0 and shadow_t + 1e-3 >= target_distance:
                                    le = _read3(light_emission, slot)
                                    pdf = source_probability * (1.0 / light_areas[slot])
                                    contribution = throughput * le * bsdf * (cos_surface * cos_light / (dist2 * ti.max(pdf, 1e-8)))
                                    radiance += contribution
                                    if depth == 0:
                                        first_bounce_direct += contribution
                        else:
                            point_index = slot - light_count
                            light_point = _read3(point_positions, point_index)
                            to_light = light_point - hit_pos
                            dist2 = ti.max(_dot(to_light, to_light), 1e-6)
                            light_dir = to_light / ti.sqrt(dist2)
                            cos_surface = ti.max(_dot(normal, light_dir), 0.0)
                            if cos_surface > 0.0:
                                shadow_origin = hit_pos + light_dir * 1e-4
                                shadow_index, shadow_t = _intersect(shadow_origin, light_dir, triangle_count, tri_a, tri_edge1, tri_edge2)
                                target_distance = ti.sqrt(dist2)
                                if shadow_index < 0 or shadow_t + 1e-3 >= target_distance:
                                    intensity = _read3(point_intensities, point_index)
                                    contribution = throughput * intensity * bsdf * (
                                        cos_surface / (dist2 * ti.max(source_probability, 1e-8))
                                    )
                                    radiance += contribution
                                    if depth == 0:
                                        first_bounce_direct += contribution

                    diffuse_weight = (diffuse.x + diffuse.y + diffuse.z) / 3.0
                    mirror_weight = (mirror.x + mirror.y + mirror.z) / 3.0
                    total_weight = diffuse_weight + mirror_weight
                    if total_weight <= 0.0:
                        active = False
                    else:
                        p_diffuse = diffuse_weight / total_weight
                        choose_diffuse = _rand(seed, pixel_index, depth, 6) < p_diffuse
                        event_probability = p_diffuse
                        weight = diffuse
                        if choose_diffuse:
                            direction = _cosine_direction(normal, _rand(seed, pixel_index, depth, 7), _rand(seed, pixel_index, depth, 8))
                            last_specular = False
                        else:
                            direction = _normalize(direction - 2.0 * _dot(direction, normal) * normal)
                            event_probability = 1.0 - p_diffuse
                            weight = mirror
                            last_specular = True
                        throughput *= weight / ti.max(event_probability, 1e-8)
                        origin = hit_pos + direction * 1e-4

                        if depth + 1 >= min_depth:
                            survive = ti.min(ti.max(ti.max(throughput.x, throughput.y, throughput.z), 0.1), 0.95)
                            if _rand(seed, pixel_index, depth, 9) > survive:
                                active = False
                            else:
                                throughput /= survive

        out_radiance[y, x, 0] = radiance.x
        out_radiance[y, x, 1] = radiance.y
        out_radiance[y, x, 2] = radiance.z
        out_direct[y, x, 0] = first_bounce_direct.x
        out_direct[y, x, 1] = first_bounce_direct.y
        out_direct[y, x, 2] = first_bounce_direct.z

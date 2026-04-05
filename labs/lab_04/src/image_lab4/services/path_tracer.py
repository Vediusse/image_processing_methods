from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

import numpy as np
import torch

from image_lab4.io.obj_loader import load_obj_triangles
from image_lab4.math.geometry import reflect, sample_cosine_weighted_hemisphere, sample_point_on_triangle, triangle_area, triangle_normal
from image_lab4.models.scene import HitRecord, Material, Ray, RenderArtifact, ResolvedTriangle, Scene, SceneConfig
from image_lab4.models.vector import EPSILON, ColorRGB, Point3, Vec3


class PathTracer:
    def __init__(self) -> None:
        self._preview_overrides = None
        cpu_threads = max(1, os.cpu_count() or 1)
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(cpu_threads)
        except RuntimeError:
            pass
        self.torch_device = self._pick_torch_device()

    def render(self, config: SceneConfig, config_path: Optional[Path] = None, strict_resolution: bool = True) -> RenderArtifact:
        scene = self._build_scene(config, strict_resolution=strict_resolution)
        radiance = self._render_torch(scene)
        radiance = self._postprocess_radiance(radiance, scene)

        display = self._tone_map(radiance, scene)
        summary = self._build_summary(scene, radiance)
        return RenderArtifact(
            radiance=radiance,
            display=display,
            summary=summary,
            scene=scene,
            config_path=config_path,
        )

    def _build_scene(self, config: SceneConfig, strict_resolution: bool = True) -> Scene:
        material_map = {item.name: item for item in config.materials}
        self._validate_materials(config.materials)
        triangles = list(config.triangles)
        for mesh in config.obj_meshes:
            triangles.extend(
                load_obj_triangles(
                    path=Path(mesh.path),
                    material_name=mesh.material_name,
                    emission=mesh.emission,
                    transform=mesh.transform,
                )
            )
        resolved = []
        for triangle in triangles:
            material = material_map[triangle.material_name]
            normal = triangle_normal(triangle.a, triangle.b, triangle.c)
            area = triangle_area(triangle.a, triangle.b, triangle.c)
            if area <= EPSILON:
                continue
            resolved.append(
                ResolvedTriangle(
                    a=triangle.a,
                    b=triangle.b,
                    c=triangle.c,
                    material=material,
                    emission=triangle.emission,
                    area=area,
                    normal=normal,
                )
            )
        lights = [triangle for triangle in resolved if triangle.emission.max_component() > 0.0]
        if not lights:
            raise ValueError("Scene must contain at least one emissive triangle light.")
        if strict_resolution and (config.render.width < 500 or config.render.height < 500):
            raise ValueError("Resolution must be at least 500x500 according to the assignment.")
        if strict_resolution and (config.render.width > 1000 or config.render.height > 1000):
            raise ValueError("Resolution must not exceed 1000x1000.")
        forward = (config.camera.target - config.camera.position).normalized()
        right = forward.cross(config.camera.up).normalized()
        up = right.cross(forward).normalized()
        triangle_a = np.array([triangle.a.to_tuple() for triangle in resolved], dtype=float)
        triangle_b = np.array([triangle.b.to_tuple() for triangle in resolved], dtype=float)
        triangle_c = np.array([triangle.c.to_tuple() for triangle in resolved], dtype=float)
        triangle_edge1 = triangle_b - triangle_a
        triangle_edge2 = triangle_c - triangle_a
        triangle_normals = np.array([triangle.normal.to_tuple() for triangle in resolved], dtype=float)
        light_indices = np.array(
            [index for index, triangle in enumerate(resolved) if triangle.emission.max_component() > 0.0],
            dtype=int,
        )
        light_weights = np.array(
            [resolved[index].area * resolved[index].emission.average() for index in light_indices],
            dtype=float,
        )
        light_probabilities = light_weights / float(light_weights.sum())
        return Scene(
            camera=config.camera,
            render=config.render,
            triangles=resolved,
            lights=lights,
            camera_forward=forward,
            camera_right=right,
            camera_up=up,
            tan_half_fov=float(np.tan(np.radians(config.camera.fov_degrees) * 0.5)),
            triangle_a=triangle_a,
            triangle_edge1=triangle_edge1,
            triangle_edge2=triangle_edge2,
            triangle_normals=triangle_normals,
            light_indices=light_indices,
            light_probabilities=light_probabilities,
        )

    def _validate_materials(self, materials) -> None:
        for material in materials:
            for diffuse_value, mirror_value in zip(material.diffuse.to_tuple(), material.mirror.to_tuple()):
                if diffuse_value < 0.0 or mirror_value < 0.0:
                    raise ValueError("Material coefficients must be non-negative.")
                if diffuse_value + mirror_value > 1.0 + 1e-8:
                    raise ValueError(
                        "Material '{0}' violates energy conservation: diffuse + mirror must be <= 1 for each channel.".format(
                            material.name
                        )
                    )

    def _generate_camera_ray(self, scene: Scene, pixel_x: int, pixel_y: int, rng) -> Ray:
        width = scene.render.width
        height = scene.render.height
        aspect = width / height
        sample_x = (pixel_x + rng.random()) / width
        sample_y = (pixel_y + rng.random()) / height
        ndc_x = (2.0 * sample_x - 1.0) * aspect * scene.tan_half_fov
        ndc_y = (1.0 - 2.0 * sample_y) * scene.tan_half_fov
        direction = (scene.camera_forward + scene.camera_right * ndc_x + scene.camera_up * ndc_y).normalized()
        return Ray(origin=scene.camera.position, direction=direction)

    def _trace_path(self, scene: Scene, ray: Ray, rng) -> np.ndarray:
        throughput = ColorRGB.one()
        radiance = ColorRGB.zero()
        last_event_specular = True

        for depth in range(scene.render.max_depth):
            hit = self._intersect_scene(scene, ray)
            if hit is None:
                radiance = radiance + throughput * scene.render.background
                break

            oriented_normal = self._face_forward_normal(hit.normal, ray.direction)
            material = hit.triangle.material

            if hit.triangle.emission.max_component() > 0.0 and last_event_specular:
                radiance = radiance + throughput * hit.triangle.emission

            if material.diffuse.max_component() > 0.0:
                direct = self._sample_direct_lighting(scene, hit, oriented_normal, rng)
                radiance = radiance + throughput * direct

            event = self._sample_material_event(material, rng)
            if event is None:
                break

            if event == "diffuse":
                next_direction = sample_cosine_weighted_hemisphere(oriented_normal, rng.random(), rng.random())
                probability = self._event_probability(material, "diffuse")
                throughput = throughput * (material.diffuse / probability)
                last_event_specular = False
            else:
                next_direction = reflect(ray.direction, oriented_normal)
                probability = self._event_probability(material, "mirror")
                throughput = throughput * (material.mirror / probability)
                last_event_specular = True

            ray = Ray(origin=hit.position + next_direction * 1e-4, direction=next_direction)

            if depth + 1 >= scene.render.min_depth:
                survive_probability = min(max(throughput.max_component(), 0.1), 0.95)
                if rng.random() > survive_probability:
                    break
                throughput = throughput / survive_probability

        return np.array(radiance.to_tuple(), dtype=float)

    def _sample_direct_lighting(self, scene: Scene, hit: HitRecord, normal: Vec3, rng) -> ColorRGB:
        if not scene.lights:
            return ColorRGB.zero()
        sampled_light_slot = int(rng.choice(len(scene.light_indices), p=scene.light_probabilities))
        triangle_index = int(scene.light_indices[sampled_light_slot])
        light = scene.triangles[triangle_index]
        light_probability = float(scene.light_probabilities[sampled_light_slot])
        light_point = sample_point_on_triangle(light.a, light.b, light.c, rng.random(), rng.random())
        to_light = light_point - hit.position
        distance_squared = max(to_light.dot(to_light), EPSILON)
        direction = to_light / np.sqrt(distance_squared)
        cosine_surface = max(normal.dot(direction), 0.0)
        cosine_light = max(light.normal.dot(direction * -1.0), 0.0)
        if cosine_surface <= 0.0 or cosine_light <= 0.0:
            return ColorRGB.zero()
        shadow_ray = Ray(origin=hit.position + direction * 1e-4, direction=direction)
        shadow_hit = self._intersect_scene(scene, shadow_ray)
        distance = float(np.sqrt(distance_squared))
        if shadow_hit is None or shadow_hit.triangle_index != triangle_index or shadow_hit.distance + 1e-3 < distance:
            return ColorRGB.zero()
        pdf_area = 1.0 / light.area
        total_pdf = light_probability * pdf_area
        bsdf = hit.triangle.material.diffuse / np.pi
        contribution = light.emission * bsdf * (cosine_surface * cosine_light / (distance_squared * total_pdf))
        return contribution

    def _sample_material_event(self, material: Material, rng):
        diffuse_weight = material.diffuse.average()
        mirror_weight = material.mirror.average()
        total = diffuse_weight + mirror_weight
        if total <= 0.0:
            return None
        if rng.random() < diffuse_weight / total:
            return "diffuse"
        return "mirror"

    def _event_probability(self, material: Material, event_name: str) -> float:
        diffuse_weight = material.diffuse.average()
        mirror_weight = material.mirror.average()
        total = diffuse_weight + mirror_weight
        if total <= 0.0:
            return 1.0
        if event_name == "diffuse":
            return diffuse_weight / total
        return mirror_weight / total

    def _intersect_scene(self, scene: Scene, ray: Ray):
        origin = np.array(ray.origin.to_tuple(), dtype=float)
        direction = np.array(ray.direction.to_tuple(), dtype=float)
        pvec = np.cross(np.broadcast_to(direction, scene.triangle_edge2.shape), scene.triangle_edge2)
        det = np.einsum("ij,ij->i", scene.triangle_edge1, pvec)
        valid = np.abs(det) > EPSILON
        if not np.any(valid):
            return None

        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]
        tvec = origin - scene.triangle_a
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        valid &= (u >= 0.0) & (u <= 1.0)
        if not np.any(valid):
            return None

        qvec = np.cross(tvec, scene.triangle_edge1)
        v = np.einsum("j,ij->i", direction, qvec) * inv_det
        valid &= (v >= 0.0) & ((u + v) <= 1.0)
        if not np.any(valid):
            return None

        t = np.einsum("ij,ij->i", scene.triangle_edge2, qvec) * inv_det
        valid &= t > EPSILON
        if not np.any(valid):
            return None

        candidate_indices = np.nonzero(valid)[0]
        distances = t[candidate_indices]
        closest_local_index = int(np.argmin(distances))
        triangle_index = int(candidate_indices[closest_local_index])
        distance = float(distances[closest_local_index])
        triangle = scene.triangles[triangle_index]
        position = ray.origin + ray.direction * distance
        normal = Vec3(*scene.triangle_normals[triangle_index])
        return HitRecord(
            distance=distance,
            position=position,
            normal=normal,
            triangle=triangle,
            triangle_index=triangle_index,
        )

    def _face_forward_normal(self, normal: Vec3, direction: Vec3) -> Vec3:
        if normal.dot(direction) < 0.0:
            return normal
        return normal * -1.0

    def _tone_map(self, radiance: np.ndarray, scene: Scene) -> np.ndarray:
        positive = np.clip(radiance, 0.0, None)
        if scene.render.normalization == "mean":
            scale = np.mean(np.clip(radiance, 0.0, None))
            scale = scale / max(scene.render.normalization_value, 1e-8)
        elif scene.render.normalization == "percentile":
            percentile = min(max(scene.render.normalization_value, 50.0), 99.9)
            scale = float(np.percentile(positive, percentile))
        else:
            scale = np.max(positive)
        scale = max(scale, 1e-8)
        normalized = np.clip(radiance / scale, 0.0, 1.0)
        gamma_corrected = np.power(normalized, 1.0 / scene.render.gamma)
        return np.clip(gamma_corrected * 255.0, 0.0, 255.0)

    def _postprocess_radiance(self, radiance: np.ndarray, scene: Scene) -> np.ndarray:
        clipped = np.clip(radiance, 0.0, None)
        firefly_limit = np.percentile(clipped, 99.4)
        clipped = np.clip(clipped, 0.0, max(firefly_limit, 1e-6))
        strength = 0.14 if scene.render.samples_per_pixel >= 8 else 0.24
        return self._edge_aware_denoise(clipped, blend=strength)

    def _edge_aware_denoise(self, image: np.ndarray, blend: float) -> np.ndarray:
        padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
        center = padded[1:-1, 1:-1]
        spatial_weights = [
            (0.06, padded[:-2, :-2]),
            (0.10, padded[:-2, 1:-1]),
            (0.06, padded[:-2, 2:]),
            (0.10, padded[1:-1, :-2]),
            (0.36, padded[1:-1, 1:-1]),
            (0.10, padded[1:-1, 2:]),
            (0.06, padded[2:, :-2]),
            (0.10, padded[2:, 1:-1]),
            (0.06, padded[2:, 2:]),
        ]
        center_luma = 0.2126 * center[:, :, 0] + 0.7152 * center[:, :, 1] + 0.0722 * center[:, :, 2]
        accum = np.zeros_like(center)
        weight_sum = np.zeros(center.shape[:2] + (1,), dtype=center.dtype)
        sigma = max(float(np.percentile(center_luma, 75)), 1e-3) * 0.35
        for spatial_weight, neighbor in spatial_weights:
            neighbor_luma = 0.2126 * neighbor[:, :, 0] + 0.7152 * neighbor[:, :, 1] + 0.0722 * neighbor[:, :, 2]
            range_weight = np.exp(-np.abs(neighbor_luma - center_luma) / sigma)[:, :, None]
            total_weight = spatial_weight * range_weight
            accum += neighbor * total_weight
            weight_sum += total_weight
        filtered = accum / np.maximum(weight_sum, 1e-8)
        return center * (1.0 - blend) + filtered * blend

    def _build_summary(self, scene: Scene, radiance: np.ndarray) -> str:
        max_radiance = float(np.max(radiance))
        mean_radiance = float(np.mean(radiance))
        return (
            "Path tracing завершен.\n"
            "Backend: {0}\n"
            "Треугольников: {1}\n"
            "Источников света: {2}\n"
            "Разрешение: {3}x{4}\n"
            "SPP: {5}\n"
            "Максимальная абсолютная яркость: {6:.6f}\n"
            "Средняя абсолютная яркость: {7:.6f}\n"
            "Нормировка: {8}, gamma={9:.2f}".format(
                self.torch_device,
                len(scene.triangles),
                len(scene.lights),
                scene.render.width,
                scene.render.height,
                scene.render.samples_per_pixel,
                max_radiance,
                mean_radiance,
                scene.render.normalization,
                scene.render.gamma,
            )
        )

    def _pick_torch_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _render_torch(self, scene: Scene) -> np.ndarray:
        device = torch.device(self.torch_device)
        width = scene.render.width
        height = scene.render.height
        pixel_count = width * height
        render = scene.render
        generator = torch.Generator(device=device)
        generator.manual_seed(render.seed)

        triangle_a = torch.tensor(scene.triangle_a, dtype=torch.float32, device=device)
        triangle_edge1 = torch.tensor(scene.triangle_edge1, dtype=torch.float32, device=device)
        triangle_edge2 = torch.tensor(scene.triangle_edge2, dtype=torch.float32, device=device)
        triangle_normals = torch.tensor(scene.triangle_normals, dtype=torch.float32, device=device)
        material_diffuse = torch.tensor([triangle.material.diffuse.to_tuple() for triangle in scene.triangles], dtype=torch.float32, device=device)
        material_mirror = torch.tensor([triangle.material.mirror.to_tuple() for triangle in scene.triangles], dtype=torch.float32, device=device)
        triangle_emission = torch.tensor([triangle.emission.to_tuple() for triangle in scene.triangles], dtype=torch.float32, device=device)
        light_indices = torch.tensor(scene.light_indices, dtype=torch.long, device=device)
        light_probabilities = torch.tensor(scene.light_probabilities, dtype=torch.float32, device=device)
        light_areas = torch.tensor([scene.triangles[int(index)].area for index in scene.light_indices], dtype=torch.float32, device=device)

        camera_position = torch.tensor(scene.camera.position.to_tuple(), dtype=torch.float32, device=device)
        camera_forward = torch.tensor(scene.camera_forward.to_tuple(), dtype=torch.float32, device=device)
        camera_right = torch.tensor(scene.camera_right.to_tuple(), dtype=torch.float32, device=device)
        camera_up = torch.tensor(scene.camera_up.to_tuple(), dtype=torch.float32, device=device)
        background = torch.tensor(render.background.to_tuple(), dtype=torch.float32, device=device)

        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        aspect = float(width) / float(height)

        radiance = torch.zeros((pixel_count, 3), dtype=torch.float32, device=device)
        tan_half_fov = float(scene.tan_half_fov)

        for _ in range(render.samples_per_pixel):
            jitter_x = torch.rand(pixel_count, generator=generator, device=device)
            jitter_y = torch.rand(pixel_count, generator=generator, device=device)
            sample_x = (xx + jitter_x) / width
            sample_y = (yy + jitter_y) / height
            ndc_x = (2.0 * sample_x - 1.0) * aspect * tan_half_fov
            ndc_y = (1.0 - 2.0 * sample_y) * tan_half_fov
            directions = camera_forward[None, :] + camera_right[None, :] * ndc_x[:, None] + camera_up[None, :] * ndc_y[:, None]
            directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True).clamp_min(1e-8)
            origins = camera_position[None, :].expand(pixel_count, 3).clone()
            throughput = torch.ones((pixel_count, 3), dtype=torch.float32, device=device)
            sample_radiance = torch.zeros((pixel_count, 3), dtype=torch.float32, device=device)
            active = torch.ones(pixel_count, dtype=torch.bool, device=device)
            last_specular = torch.ones(pixel_count, dtype=torch.bool, device=device)

            for depth in range(render.max_depth):
                if not bool(active.any()):
                    break
                active_indices = torch.nonzero(active, as_tuple=False).squeeze(1)
                hit_mask, distances, tri_indices = self._intersect_rays_torch(
                    triangle_a,
                    triangle_edge1,
                    triangle_edge2,
                    origins[active_indices],
                    directions[active_indices],
                )
                miss_indices = active_indices[~hit_mask]
                if miss_indices.numel() > 0:
                    sample_radiance[miss_indices] += throughput[miss_indices] * background
                    active[miss_indices] = False

                hit_indices = active_indices[hit_mask]
                if hit_indices.numel() == 0:
                    continue

                hit_distances = distances[hit_mask]
                hit_tri_indices = tri_indices[hit_mask]
                positions = origins[hit_indices] + directions[hit_indices] * hit_distances[:, None]
                normals = triangle_normals[hit_tri_indices]
                oriented_normals = torch.where(
                    (normals * directions[hit_indices]).sum(dim=1, keepdim=True) < 0.0,
                    normals,
                    -normals,
                )
                emissions = triangle_emission[hit_tri_indices]
                specular_hits = last_specular[hit_indices]
                emissive_mask = emissions.max(dim=1).values > 0.0
                if bool((emissive_mask & specular_hits).any()):
                    emissive_indices = hit_indices[emissive_mask & specular_hits]
                    sample_radiance[emissive_indices] += throughput[emissive_indices] * emissions[emissive_mask & specular_hits]

                diffuse = material_diffuse[hit_tri_indices]
                mirror = material_mirror[hit_tri_indices]
                diffuse_mask = diffuse.max(dim=1).values > 0.0
                if bool(diffuse_mask.any()):
                    direct = self._sample_direct_lighting_torch(
                        scene,
                        triangle_a,
                        triangle_edge1,
                        triangle_edge2,
                        triangle_normals,
                        triangle_emission,
                        diffuse,
                        light_indices,
                        light_probabilities,
                        light_areas,
                        hit_tri_indices,
                        positions,
                        oriented_normals,
                        throughput[hit_indices],
                        generator,
                    )
                    sample_radiance[hit_indices] += direct

                next_dirs, event_probabilities, is_specular = self._sample_events_torch(
                    diffuse,
                    mirror,
                    oriented_normals,
                    directions[hit_indices],
                    generator,
                )
                valid_events = event_probabilities > 0.0
                invalid_indices = hit_indices[~valid_events]
                if invalid_indices.numel() > 0:
                    active[invalid_indices] = False

                hit_indices = hit_indices[valid_events]
                if hit_indices.numel() == 0:
                    continue
                next_dirs = next_dirs[valid_events]
                event_probabilities = event_probabilities[valid_events]
                is_specular = is_specular[valid_events]
                selected_diffuse = diffuse[valid_events]
                selected_mirror = mirror[valid_events]
                weights = torch.where(is_specular[:, None], selected_mirror, selected_diffuse)
                throughput[hit_indices] = throughput[hit_indices] * (weights / event_probabilities[:, None].clamp_min(1e-8))
                origins[hit_indices] = positions[valid_events] + next_dirs * 1e-4
                directions[hit_indices] = next_dirs
                last_specular[hit_indices] = is_specular

                if depth + 1 >= render.min_depth:
                    survive = throughput[hit_indices].max(dim=1).values.clamp(0.1, 0.95)
                    keep = torch.rand(hit_indices.numel(), generator=generator, device=device) <= survive
                    dropped = hit_indices[~keep]
                    if dropped.numel() > 0:
                        active[dropped] = False
                    kept = hit_indices[keep]
                    if kept.numel() > 0:
                        throughput[kept] = throughput[kept] / survive[keep][:, None]

            radiance += sample_radiance

        radiance = radiance / max(render.samples_per_pixel, 1)
        return radiance.reshape(height, width, 3).detach().cpu().numpy()

    def _intersect_rays_torch(self, triangle_a, triangle_edge1, triangle_edge2, origins, directions):
        pvec = torch.cross(directions[:, None, :].expand(-1, triangle_edge2.shape[0], -1), triangle_edge2[None, :, :], dim=2)
        det = (triangle_edge1[None, :, :] * pvec).sum(dim=2)
        valid = det.abs() > EPSILON
        inv_det = torch.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]
        tvec = origins[:, None, :] - triangle_a[None, :, :]
        u = (tvec * pvec).sum(dim=2) * inv_det
        valid = valid & (u >= 0.0) & (u <= 1.0)
        qvec = torch.cross(tvec, triangle_edge1[None, :, :], dim=2)
        v = (directions[:, None, :] * qvec).sum(dim=2) * inv_det
        valid = valid & (v >= 0.0) & ((u + v) <= 1.0)
        t = (triangle_edge2[None, :, :] * qvec).sum(dim=2) * inv_det
        valid = valid & (t > EPSILON)
        masked_t = torch.where(valid, t, torch.full_like(t, float("inf")))
        distances, tri_indices = masked_t.min(dim=1)
        hit_mask = torch.isfinite(distances)
        return hit_mask, distances, tri_indices

    def _sample_events_torch(self, diffuse, mirror, normals, incoming, generator):
        diffuse_weight = diffuse.mean(dim=1)
        mirror_weight = mirror.mean(dim=1)
        total = diffuse_weight + mirror_weight
        probabilities = torch.where(total > 0.0, diffuse_weight / total.clamp_min(1e-8), torch.zeros_like(total))
        choose_diffuse = torch.rand(total.shape[0], generator=generator, device=total.device) < probabilities
        u1 = torch.rand(total.shape[0], generator=generator, device=total.device)
        u2 = torch.rand(total.shape[0], generator=generator, device=total.device)
        diffuse_dirs = self._sample_cosine_dirs_torch(normals, u1, u2)
        reflected = incoming - 2.0 * (incoming * normals).sum(dim=1, keepdim=True) * normals
        reflected = reflected / torch.linalg.norm(reflected, dim=1, keepdim=True).clamp_min(1e-8)
        next_dirs = torch.where(choose_diffuse[:, None], diffuse_dirs, reflected)
        event_prob = torch.where(choose_diffuse, probabilities, 1.0 - probabilities)
        return next_dirs, event_prob, ~choose_diffuse

    def _sample_cosine_dirs_torch(self, normals, u1, u2):
        r = torch.sqrt(u1.clamp(0.0, 1.0))
        phi = 2.0 * np.pi * u2
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = torch.sqrt((1.0 - u1).clamp_min(0.0))
        tangent = torch.zeros_like(normals)
        use_y = normals[:, 0].abs() > 0.1
        tangent[use_y] = torch.nn.functional.normalize(torch.cross(torch.tensor([0.0, 1.0, 0.0], device=normals.device).expand(use_y.sum(), 3), normals[use_y], dim=1), dim=1)
        tangent[~use_y] = torch.nn.functional.normalize(torch.cross(torch.tensor([1.0, 0.0, 0.0], device=normals.device).expand((~use_y).sum(), 3), normals[~use_y], dim=1), dim=1)
        bitangent = torch.nn.functional.normalize(torch.cross(normals, tangent, dim=1), dim=1)
        dirs = tangent * x[:, None] + bitangent * y[:, None] + normals * z[:, None]
        return torch.nn.functional.normalize(dirs, dim=1)

    def _sample_direct_lighting_torch(
        self,
        scene,
        triangle_a,
        triangle_edge1,
        triangle_edge2,
        triangle_normals,
        triangle_emission,
        diffuse,
        light_indices,
        light_probabilities,
        light_areas,
        hit_tri_indices,
        positions,
        normals,
        throughput,
        generator,
    ):
        ray_count = positions.shape[0]
        sampled_slots = torch.multinomial(light_probabilities, ray_count, replacement=True, generator=generator)
        sampled_triangle_indices = light_indices[sampled_slots]
        selected_a = triangle_a[sampled_triangle_indices]
        selected_b = selected_a + triangle_edge1[sampled_triangle_indices]
        selected_c = selected_a + triangle_edge2[sampled_triangle_indices]
        u1 = torch.rand(ray_count, generator=generator, device=positions.device)
        u2 = torch.rand(ray_count, generator=generator, device=positions.device)
        su1 = torch.sqrt(u1)
        alpha = 1.0 - su1
        beta = su1 * (1.0 - u2)
        gamma = su1 * u2
        light_points = selected_a * alpha[:, None] + selected_b * beta[:, None] + selected_c * gamma[:, None]
        to_light = light_points - positions
        distance_squared = (to_light * to_light).sum(dim=1).clamp_min(EPSILON)
        direction = to_light / torch.sqrt(distance_squared)[:, None]
        cosine_surface = (normals * direction).sum(dim=1).clamp_min(0.0)
        light_normals = triangle_normals[sampled_triangle_indices]
        cosine_light = (light_normals * (-direction)).sum(dim=1).clamp_min(0.0)
        visible = (cosine_surface > 0.0) & (cosine_light > 0.0)
        if not bool(visible.any()):
            return torch.zeros_like(throughput)
        shadow_origins = positions + direction * 1e-4
        shadow_hit_mask, shadow_distances, shadow_tri_indices = self._intersect_rays_torch(
            triangle_a,
            triangle_edge1,
            triangle_edge2,
            shadow_origins,
            direction,
        )
        target_distance = torch.sqrt(distance_squared)
        visible = visible & shadow_hit_mask & (shadow_tri_indices == sampled_triangle_indices) & (shadow_distances + 1e-3 >= target_distance)
        if not bool(visible.any()):
            return torch.zeros_like(throughput)
        bsdf = diffuse / np.pi
        pdf = light_probabilities[sampled_slots] * (1.0 / light_areas[sampled_slots])
        contrib = triangle_emission[sampled_triangle_indices] * bsdf * (
            cosine_surface[:, None] * cosine_light[:, None] / (distance_squared[:, None] * pdf[:, None].clamp_min(1e-8))
        )
        contrib = torch.where(visible[:, None], contrib, torch.zeros_like(contrib))
        return throughput * contrib

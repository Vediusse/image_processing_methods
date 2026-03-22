from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from image_lab1.math.geometry import calculate_triangle_normal
from image_lab1.models.scene import ColorRGB, Scenario
from image_lab1.models.vector import Point3, Vector3
from image_lab1.services.scenario_runner import ScenarioRunner


@dataclass(frozen=True)
class ObserverRenderResult:
    image: np.ndarray
    hit_pixels: int
    resolution: tuple[int, int]
    viewport_label: str


class ObserverRenderer:
    def __init__(self, width: int = 640, height: int = 640, fov_degrees: float = 36.0) -> None:
        self.width = width
        self.height = height
        self.fov_degrees = fov_degrees
        self.runner = ScenarioRunner()

    def render(self, scenario: Scenario) -> ObserverRenderResult:
        normal = calculate_triangle_normal(scenario.triangle)
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        centroid = (scenario.triangle.a + scenario.triangle.b + scenario.triangle.c) / 3.0
        camera_position = scenario.observer.position
        forward = (centroid - camera_position).normalized()
        world_up = Vector3(0.0, 0.0, 1.0)
        if abs(forward.dot(world_up)) > 0.95:
            world_up = Vector3(0.0, 1.0, 0.0)
        right = forward.cross(world_up).normalized()
        up = right.cross(forward).normalized()

        projected = [
            self._project(point, camera_position, right, up, forward)
            for point in (scenario.triangle.a, scenario.triangle.b, scenario.triangle.c)
        ]
        if any(item is None for item in projected):
            raise ValueError("Треугольник не попадает в поле зрения наблюдателя.")
        vertices_2d = [item for item in projected if item is not None]

        min_x = max(int(math.floor(min(p[0] for p in vertices_2d))) - 1, 0)
        max_x = min(int(math.ceil(max(p[0] for p in vertices_2d))) + 1, self.width - 1)
        min_y = max(int(math.floor(min(p[1] for p in vertices_2d))) - 1, 0)
        max_y = min(int(math.ceil(max(p[1] for p in vertices_2d))) + 1, self.height - 1)

        plane_point = scenario.triangle.a
        plane_normal = normal
        half_tan = math.tan(math.radians(self.fov_degrees) / 2.0)
        aspect_ratio = self.width / self.height
        hit_pixels = 0

        for pixel_y in range(min_y, max_y + 1):
            for pixel_x in range(min_x, max_x + 1):
                sample_x = pixel_x + 0.5
                sample_y = pixel_y + 0.5
                if not self._inside_screen_triangle(sample_x, sample_y, vertices_2d):
                    continue

                ndc_x = ((sample_x / self.width) * 2.0 - 1.0) * aspect_ratio * half_tan
                ndc_y = (1.0 - (sample_y / self.height) * 2.0) * half_tan
                ray_direction = (forward + right * ndc_x + up * ndc_y).normalized()
                point = self._intersect_triangle(
                    ray_origin=camera_position,
                    ray_direction=ray_direction,
                    plane_point=plane_point,
                    plane_normal=plane_normal,
                    scenario=scenario,
                )
                if point is None:
                    continue

                evaluation = self.runner.evaluate_surface_point(
                    scenario=scenario,
                    global_point=point,
                    point_name="pixel",
                    normal=normal,
                )
                image[pixel_y, pixel_x] = self._color_to_rgb8(evaluation.total_brightness)
                hit_pixels += 1

        return ObserverRenderResult(
            image=image,
            hit_pixels=hit_pixels,
            resolution=(self.width, self.height),
            viewport_label=(
                "Вид из положения наблюдателя на центр треугольника, "
                f"FOV = {self.fov_degrees:.1f}°, разрешение = {self.width}x{self.height}"
            ),
        )

    def _project(
        self,
        point: Point3,
        camera_position: Point3,
        right: Vector3,
        up: Vector3,
        forward: Vector3,
    ) -> Optional[tuple[float, float]]:
        relative = point - camera_position
        x_camera = relative.dot(right)
        y_camera = relative.dot(up)
        z_camera = relative.dot(forward)
        if z_camera <= 1e-6:
            return None

        focal = (self.height / 2.0) / math.tan(math.radians(self.fov_degrees) / 2.0)
        screen_x = self.width / 2.0 + (x_camera / z_camera) * focal
        screen_y = self.height / 2.0 - (y_camera / z_camera) * focal
        return (screen_x, screen_y)

    def _inside_screen_triangle(
        self,
        px: float,
        py: float,
        vertices_2d: list[tuple[float, float]],
    ) -> bool:
        (x1, y1), (x2, y2), (x3, y3) = vertices_2d
        denominator = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        if abs(denominator) <= 1e-9:
            return False
        a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denominator
        b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denominator
        c = 1.0 - a - b
        tolerance = 1e-5
        return a >= -tolerance and b >= -tolerance and c >= -tolerance

    def _intersect_triangle(
        self,
        ray_origin: Point3,
        ray_direction: Vector3,
        plane_point: Point3,
        plane_normal: Vector3,
        scenario: Scenario,
    ) -> Optional[Point3]:
        denominator = plane_normal.dot(ray_direction)
        if abs(denominator) <= 1e-9:
            return None
        distance = plane_normal.dot(plane_point - ray_origin) / denominator
        if distance <= 0:
            return None
        point = ray_origin + ray_direction * distance
        if not self._is_inside_world_triangle(point, scenario):
            return None
        return point

    def _is_inside_world_triangle(self, point: Point3, scenario: Scenario) -> bool:
        a = scenario.triangle.a
        b = scenario.triangle.b
        c = scenario.triangle.c
        v0 = b - a
        v1 = c - a
        v2 = point - a
        dot00 = v0.dot(v0)
        dot01 = v0.dot(v1)
        dot02 = v0.dot(v2)
        dot11 = v1.dot(v1)
        dot12 = v1.dot(v2)
        denominator = dot00 * dot11 - dot01 * dot01
        if abs(denominator) <= 1e-9:
            return False
        u = (dot11 * dot02 - dot01 * dot12) / denominator
        v = (dot00 * dot12 - dot01 * dot02) / denominator
        tolerance = 1e-5
        return u >= -tolerance and v >= -tolerance and (u + v) <= 1.0 + tolerance

    def _color_to_rgb8(self, color: ColorRGB) -> np.ndarray:
        source = np.array([color.r, color.g, color.b], dtype=np.float64)
        mapped = np.power(source / (1.0 + source), 1.0 / 2.2)
        mapped = np.clip(mapped, 0.0, 1.0)
        return (mapped * 255.0).astype(np.uint8)

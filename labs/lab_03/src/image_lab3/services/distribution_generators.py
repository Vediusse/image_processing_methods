from __future__ import annotations

import numpy as np

from image_lab3.math.geometry import build_orthonormal_basis
from image_lab3.models.config import CircleConfig, TriangleConfig
from image_lab3.models.vector import Vector3


class DistributionGenerators:
    def triangle_points(self, triangle: TriangleConfig, sample_count: int, rng: np.random.Generator):
        u1 = rng.random(sample_count)
        u2 = rng.random(sample_count)
        su1 = np.sqrt(u1)
        alpha = 1.0 - su1
        beta = su1 * (1.0 - u2)
        gamma = su1 * u2
        a = np.array(triangle.a.to_tuple())
        b = np.array(triangle.b.to_tuple())
        c = np.array(triangle.c.to_tuple())
        points = alpha[:, None] * a + beta[:, None] * b + gamma[:, None] * c
        auxiliary = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "u1_recovered": u1,
            "u2_recovered": u2,
        }
        return points, auxiliary

    def circle_points(self, circle: CircleConfig, sample_count: int, rng: np.random.Generator):
        tangent, bitangent, normal = build_orthonormal_basis(circle.normal)
        r = circle.radius * np.sqrt(rng.random(sample_count))
        phi = 2.0 * np.pi * rng.random(sample_count)
        local_x = r * np.cos(phi)
        local_y = r * np.sin(phi)
        center = np.array(circle.center.to_tuple())
        t = np.array(tangent.to_tuple())
        b = np.array(bitangent.to_tuple())
        points = center + local_x[:, None] * t + local_y[:, None] * b
        auxiliary = {
            "radius_squared_normalized": (r / circle.radius) ** 2,
            "distances": r,
            "phi": phi,
            "local_x": local_x,
            "local_y": local_y,
            "radius": circle.radius,
            "normal": np.array(normal.to_tuple()),
        }
        return points, auxiliary

    def sphere_directions(self, sample_count: int, rng: np.random.Generator):
        z = 1.0 - 2.0 * rng.random(sample_count)
        phi = 2.0 * np.pi * rng.random(sample_count)
        radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)
        points = np.column_stack((x, y, z))
        auxiliary = {"z": z}
        return points, auxiliary

    def cosine_weighted_directions(self, normal: Vector3, sample_count: int, rng: np.random.Generator):
        tangent, bitangent, n = build_orthonormal_basis(normal)
        r = np.sqrt(rng.random(sample_count))
        phi = 2.0 * np.pi * rng.random(sample_count)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.sqrt(np.maximum(1.0 - x * x - y * y, 0.0))
        t = np.array(tangent.to_tuple())
        b = np.array(bitangent.to_tuple())
        n_arr = np.array(n.to_tuple())
        points = x[:, None] * t + y[:, None] * b + z[:, None] * n_arr
        cosine = points @ n_arr
        auxiliary = {"cosine": cosine, "normal": n_arr}
        return points, auxiliary

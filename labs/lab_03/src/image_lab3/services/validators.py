from __future__ import annotations

import numpy as np

from image_lab3.models.results import DistributionMetrics
from image_lab3.report.uniformity_checks import rectangle_counts, relative_spread


class DistributionValidators:
    def triangle_metrics(self, points, auxiliary, triangle_vertices) -> DistributionMetrics:
        alpha = auxiliary["alpha"]
        beta = auxiliary["beta"]
        gamma = auxiliary["gamma"]
        inside = np.logical_and.reduce((alpha >= -1e-9, beta >= -1e-9, gamma >= -1e-9))
        normal = np.cross(triangle_vertices[1] - triangle_vertices[0], triangle_vertices[2] - triangle_vertices[0])
        normal = normal / np.linalg.norm(normal)
        plane_error = float(
            np.mean(np.abs((points - triangle_vertices[0]) @ normal))
        )
        centroid = sum(triangle_vertices) / 3.0
        centroid_error = float(np.linalg.norm(points.mean(axis=0) - centroid))
        _, counts = rectangle_counts(beta, gamma, auxiliary["proof_rectangles"])
        uniformity = relative_spread(counts)
        return DistributionMetrics(
            title="Равномерные точки в треугольнике",
            inside_ratio=float(np.mean(inside)),
            norm_error=plane_error,
            centroid_or_mean_error=centroid_error,
            uniformity_score=uniformity,
            note="Равномерность проверяется новым методом: в локальных координатах (u, v) автоматически строятся одинаковые прямоугольники внутри треугольника, после чего сравнивается число точек в каждом из них. Если счётчики близки, распределение считаем равномерным на глазок.",
        )

    def circle_metrics(self, points, auxiliary, center, radius) -> DistributionMetrics:
        distances = np.linalg.norm(points - center, axis=1)
        plane_projection = np.abs((points - center) @ auxiliary["normal"])
        inside = np.logical_and(distances <= radius + 1e-9, plane_projection <= 1e-9)
        _, counts = rectangle_counts(auxiliary["local_x"], auxiliary["local_y"], auxiliary["proof_rectangles"])
        uniformity = relative_spread(counts)
        centroid_error = float(np.linalg.norm(points.mean(axis=0) - center))
        return DistributionMetrics(
            title="Равномерные точки в круге",
            inside_ratio=float(np.mean(inside)),
            norm_error=float(np.mean(plane_projection)),
            centroid_or_mean_error=centroid_error,
            uniformity_score=uniformity,
            note="Равномерность проверяется новым методом: внутри круга автоматически строятся одинаковые прямоугольники в локальных координатах, затем сравнивается число точек в каждом прямоугольнике. Если значения близки, распределение считаем равномерным на глазок.",
        )

    def sphere_metrics(self, points, auxiliary) -> DistributionMetrics:
        norms = np.linalg.norm(points, axis=1)
        z = auxiliary["z"]
        hist, edges = np.histogram(z, bins=20, range=(-1.0, 1.0), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        target = 0.5 * np.ones_like(centers)
        uniformity = float(np.sqrt(np.mean((hist - target) ** 2)))
        return DistributionMetrics(
            title="Равномерные направления на сфере",
            inside_ratio=1.0,
            norm_error=float(np.mean(np.abs(norms - 1.0))),
            centroid_or_mean_error=float(np.linalg.norm(points.mean(axis=0))),
            uniformity_score=uniformity,
            note="Равномерность на сфере подтверждается единичной длиной всех направлений и сравнением распределения z с теоретической постоянной плотностью 1/2 на [-1, 1].",
        )

    def cosine_metrics(self, points, auxiliary) -> DistributionMetrics:
        norms = np.linalg.norm(points, axis=1)
        cosine = auxiliary["cosine"]
        hemisphere = cosine >= -1e-9
        hist, edges = np.histogram(cosine, bins=12, range=(0.0, 1.0), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        target = 2.0 * centers
        rmse = float(np.sqrt(np.mean((hist - target) ** 2)))
        return DistributionMetrics(
            title="Косинусное распределение направлений",
            inside_ratio=float(np.mean(hemisphere)),
            norm_error=float(np.mean(np.abs(norms - 1.0))),
            centroid_or_mean_error=float(abs(np.mean(cosine) - (2.0 / 3.0))),
            uniformity_score=rmse,
            note="Косинусный закон подтверждается тем, что μ = cos(θ) имеет теоретическую плотность 2μ на [0, 1], а все сформированные направления лежат в верхней полусфере относительно нормали.",
        )

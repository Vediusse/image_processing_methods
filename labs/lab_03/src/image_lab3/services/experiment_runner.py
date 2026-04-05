from __future__ import annotations

import numpy as np

from image_lab3.models.results import DistributionResult, ExperimentResult
from image_lab3.report.uniformity_checks import circle_rectangles, triangle_rectangles
from image_lab3.services.distribution_generators import DistributionGenerators
from image_lab3.services.validators import DistributionValidators


class ExperimentRunner:
    def __init__(self) -> None:
        self.generators = DistributionGenerators()
        self.validators = DistributionValidators()

    def run(self, config) -> ExperimentResult:
        if config.sample_count <= 0:
            raise ValueError("Число выборок должно быть положительным.")
        if config.circle.radius <= 0:
            raise ValueError("Радиус круга должен быть положительным.")

        rng = np.random.default_rng(config.seed)
        distributions = {}

        triangle_points, triangle_aux = self.generators.triangle_points(config.triangle, config.sample_count, rng)
        triangle_aux["proof_rectangles"] = triangle_rectangles(config.uniformity_rectangle_count)
        triangle_aux["proof_rectangle_count"] = config.uniformity_rectangle_count
        triangle_vertices = np.array([
            config.triangle.a.to_tuple(),
            config.triangle.b.to_tuple(),
            config.triangle.c.to_tuple(),
        ])
        distributions["triangle"] = DistributionResult(
            key="triangle",
            title="Треугольник",
            points=triangle_points,
            metrics=self.validators.triangle_metrics(triangle_points, triangle_aux, triangle_vertices),
            auxiliary=triangle_aux,
        )

        circle_points, circle_aux = self.generators.circle_points(config.circle, config.sample_count, rng)
        circle_aux["proof_rectangles"] = circle_rectangles(config.circle.radius, config.uniformity_rectangle_count)
        circle_aux["proof_rectangle_count"] = config.uniformity_rectangle_count
        distributions["circle"] = DistributionResult(
            key="circle",
            title="Круг",
            points=circle_points,
            metrics=self.validators.circle_metrics(
                circle_points,
                circle_aux,
                np.array(config.circle.center.to_tuple()),
                config.circle.radius,
            ),
            auxiliary=circle_aux,
        )

        sphere_points, sphere_aux = self.generators.sphere_directions(config.sample_count, rng)
        distributions["sphere"] = DistributionResult(
            key="sphere",
            title="Сфера",
            points=sphere_points,
            metrics=self.validators.sphere_metrics(sphere_points, sphere_aux),
            auxiliary=sphere_aux,
        )

        cosine_points, cosine_aux = self.generators.cosine_weighted_directions(
            config.cosine_normal, config.sample_count, rng
        )
        distributions["cosine"] = DistributionResult(
            key="cosine",
            title="Косинусное распределение",
            points=cosine_points,
            metrics=self.validators.cosine_metrics(cosine_points, cosine_aux),
            auxiliary=cosine_aux,
        )

        return ExperimentResult(sample_count=config.sample_count, distributions=distributions)

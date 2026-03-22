from __future__ import annotations

import math

from image_lab1.math.brdf import calculate_brdf
from image_lab1.math.geometry import calculate_triangle_normal, local_to_global
from image_lab1.math.optics import calculate_colored_irradiance
from image_lab1.models.results import BrightnessResult, PointEvaluation, SourceContribution
from image_lab1.models.scene import ColorRGB, LocalPoint, Scenario
from image_lab1.models.vector import Point3


class ScenarioRunner:
    def run(self, scenario: Scenario) -> BrightnessResult:
        self._validate_scenario(scenario)
        normal = calculate_triangle_normal(scenario.triangle)
        local_evaluations = [
            self._evaluate_from_local_point(scenario, point, normal) for point in scenario.local_points
        ]
        global_evaluations = [
            self._evaluate_global_point(
                scenario,
                point_name,
                scenario.global_points[point_name],
                self._find_local_coordinates(point_name, scenario.local_points),
                normal,
            )
            for point_name in scenario.global_points
        ]
        return BrightnessResult(
            local_evaluations=local_evaluations,
            global_evaluations=global_evaluations,
        )

    def _validate_scenario(self, scenario: Scenario) -> None:
        if not scenario.lights:
            raise ValueError("Scenario must contain at least one light source.")
        if scenario.material.diffuse_coefficient < 0 or scenario.material.specular_coefficient < 0:
            raise ValueError("Material reflection coefficients must be non-negative.")
        if scenario.material.shininess < 0:
            raise ValueError("Material shininess must be non-negative.")
        missing_globals = {point.name for point in scenario.local_points} - set(scenario.global_points.keys())
        if missing_globals:
            raise ValueError(
                "Global points must be provided for every local point. Missing: "
                + ", ".join(sorted(missing_globals))
            )
        for point in scenario.local_points:
            expected_global = local_to_global(scenario.triangle, point.u, point.v)
            actual_global = scenario.global_points[point.name]
            if not expected_global.almost_equal(actual_global, tolerance=1e-5):
                raise ValueError(
                    "Global point '{name}' does not match local coordinates. "
                    "Expected {expected}, got {actual}.".format(
                        name=point.name,
                        expected=expected_global.to_tuple(),
                        actual=actual_global.to_tuple(),
                    )
                )

    def _find_local_coordinates(
        self,
        point_name: str,
        local_points: list[LocalPoint],
    ) -> tuple[float, float]:
        for point in local_points:
            if point.name == point_name:
                return (point.u, point.v)
        raise ValueError(f"Missing local coordinates for point '{point_name}'.")

    def _evaluate_from_local_point(
        self,
        scenario: Scenario,
        point: LocalPoint,
        normal,
    ) -> PointEvaluation:
        global_point = local_to_global(scenario.triangle, point.u, point.v)
        return self._evaluate_global_point(
            scenario,
            point.name,
            global_point,
            (point.u, point.v),
            normal,
        )

    def evaluate_surface_point(
        self,
        scenario: Scenario,
        global_point: Point3,
        point_name: str = "render",
        local_coordinates: tuple[float, float] = (0.0, 0.0),
        normal=None,
    ) -> PointEvaluation:
        surface_normal = normal if normal is not None else calculate_triangle_normal(scenario.triangle)
        return self._evaluate_global_point(
            scenario=scenario,
            point_name=point_name,
            global_point=global_point,
            local_coordinates=local_coordinates,
            normal=surface_normal,
        )

    def _evaluate_global_point(
        self,
        scenario: Scenario,
        point_name: str,
        global_point: Point3,
        local_coordinates: tuple[float, float],
        normal,
    ) -> PointEvaluation:
        observer_direction = (scenario.observer.position - global_point).normalized()
        contributions: list[SourceContribution] = []
        total_irradiance = ColorRGB.black()
        total_brightness = ColorRGB.black()

        for light in scenario.lights:
            direction_to_light = light.position - global_point
            irradiance, distance, theta, alpha, colored_intensity, profile_value = calculate_colored_irradiance(
                light, direction_to_light, normal
            )
            brdf_value = calculate_brdf(scenario.material, normal, direction_to_light, observer_direction)
            brightness = irradiance * brdf_value * scenario.material.color
            contribution = SourceContribution(
                light_name=light.name,
                direction_to_light=direction_to_light.normalized(),
                distance=distance,
                theta_degrees=math.degrees(theta),
                alpha_degrees=math.degrees(alpha),
                angular_profile=profile_value,
                colored_intensity=colored_intensity,
                colored_irradiance=irradiance,
                brdf_value=brdf_value,
                brightness=brightness,
            )
            contributions.append(contribution)
            total_irradiance = total_irradiance + irradiance
            total_brightness = total_brightness + brightness

        return PointEvaluation(
            point_name=point_name,
            local_coordinates=local_coordinates,
            global_point=global_point,
            normal=normal,
            observer_direction=observer_direction,
            total_irradiance=total_irradiance,
            total_brightness=total_brightness,
            contributions=contributions,
        )

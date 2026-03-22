from __future__ import annotations

from dataclasses import dataclass

from .scene import ColorRGB
from .vector import Point3, Vector3


@dataclass(frozen=True)
class SourceContribution:
    light_name: str
    direction_to_light: Vector3
    distance: float
    theta_degrees: float
    alpha_degrees: float
    angular_profile: float
    colored_intensity: ColorRGB
    colored_irradiance: ColorRGB
    brdf_value: float
    brightness: ColorRGB


@dataclass(frozen=True)
class PointEvaluation:
    point_name: str
    local_coordinates: tuple[float, float]
    global_point: Point3
    normal: Vector3
    observer_direction: Vector3
    total_irradiance: ColorRGB
    total_brightness: ColorRGB
    contributions: list[SourceContribution]


@dataclass(frozen=True)
class BrightnessResult:
    local_evaluations: list[PointEvaluation]
    global_evaluations: list[PointEvaluation]

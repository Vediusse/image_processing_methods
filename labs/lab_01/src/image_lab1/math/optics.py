from __future__ import annotations

import math

from image_lab1.models.scene import ColorRGB, Light
from image_lab1.models.vector import Vector3


def calculate_axis_angle(light_axis: Vector3, source_to_point_direction: Vector3) -> float:
    axis = light_axis.normalized()
    direction = source_to_point_direction.normalized()
    cosine = max(-1.0, min(1.0, axis.dot(direction)))
    return math.acos(cosine)


def evaluate_emission_profile(theta_radians: float, exponent: float, cutoff_degrees: float) -> float:
    cutoff_radians = math.radians(cutoff_degrees)
    if theta_radians > cutoff_radians:
        return 0.0
    return max(math.cos(theta_radians), 0.0) ** exponent


def calculate_colored_irradiance(
    light: Light,
    direction_point_to_light: Vector3,
    surface_normal: Vector3,
) -> tuple[ColorRGB, float, float, float, ColorRGB, float]:
    distance = direction_point_to_light.length()
    light_to_point_direction = (direction_point_to_light * -1.0).normalized()
    source_to_point_direction = light_to_point_direction
    theta = calculate_axis_angle(light.axis_direction, source_to_point_direction)
    profile_value = evaluate_emission_profile(
        theta,
        light.profile.exponent,
        light.profile.cutoff_degrees,
    )
    colored_intensity = light.intensity * profile_value
    direction = direction_point_to_light.normalized()
    cosine_alpha = max(surface_normal.normalized().dot(direction), 0.0)
    alpha = math.acos(max(-1.0, min(1.0, cosine_alpha)))
    irradiance = colored_intensity * (cosine_alpha / (distance**2))
    return irradiance, distance, theta, alpha, colored_intensity, profile_value

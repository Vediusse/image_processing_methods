from __future__ import annotations

import math

from image_lab1.models.scene import Material
from image_lab1.models.vector import Vector3


def calculate_half_vector(direction_to_light: Vector3, direction_to_observer: Vector3) -> Vector3:
    summed = direction_to_light.normalized() + direction_to_observer.normalized()
    if summed.length() <= 1e-9:
        raise ValueError("Cannot calculate half vector for opposite light and observer directions.")
    return summed.normalized()


def calculate_brdf(
    material: Material,
    surface_normal: Vector3,
    direction_to_light: Vector3,
    direction_to_observer: Vector3,
) -> float:
    half_vector = calculate_half_vector(direction_to_light, direction_to_observer)
    n_dot_h = max(surface_normal.normalized().dot(half_vector), 0.0)
    diffuse = material.diffuse_coefficient / math.pi
    specular = material.specular_coefficient * ((material.shininess + 2.0) / (2.0 * math.pi)) * (
        n_dot_h ** material.shininess
    )
    return diffuse + specular

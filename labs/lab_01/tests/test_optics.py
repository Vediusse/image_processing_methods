import math

from image_lab1.math.brdf import calculate_brdf, calculate_half_vector
from image_lab1.math.optics import calculate_colored_irradiance
from image_lab1.models.scene import ColorRGB, Light, Material, RadiationProfile
from image_lab1.models.vector import Point3, Vector3


def test_half_vector_is_normalized() -> None:
    half_vector = calculate_half_vector(Vector3(0.0, 0.0, 1.0), Vector3(0.0, 1.0, 1.0))
    assert math.isclose(half_vector.length(), 1.0, rel_tol=1e-6)


def test_irradiance_is_positive_for_front_facing_surface() -> None:
    light = Light(
        name="L1",
        position=Point3(0.0, 0.0, 1.0),
        axis_direction=Vector3(0.0, 0.0, -1.0),
        intensity=ColorRGB(1.0, 1.0, 1.0),
        profile=RadiationProfile(exponent=1.0, cutoff_degrees=90.0),
    )
    irradiance, distance, theta, alpha, _, profile_value = calculate_colored_irradiance(
        light,
        direction_point_to_light=Vector3(0.0, 0.0, 1.0),
        surface_normal=Vector3(0.0, 0.0, 1.0),
    )
    assert irradiance.r > 0
    assert math.isclose(distance, 1.0)
    assert theta >= 0
    assert alpha >= 0
    assert profile_value > 0


def test_brdf_contains_diffuse_and_specular_terms() -> None:
    material = Material(
        color=ColorRGB(1.0, 1.0, 1.0),
        diffuse_coefficient=0.8,
        specular_coefficient=0.2,
        shininess=10.0,
    )
    value = calculate_brdf(
        material,
        surface_normal=Vector3(0.0, 0.0, 1.0),
        direction_to_light=Vector3(0.0, 0.0, 1.0),
        direction_to_observer=Vector3(0.0, 0.0, 1.0),
    )
    assert value > (0.8 / math.pi)

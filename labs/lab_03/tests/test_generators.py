import numpy as np

from image_lab3.io.config_loader import load_config
from image_lab3.services.distribution_generators import DistributionGenerators


def test_triangle_generator_produces_correct_count() -> None:
    config = load_config("examples/default_config.json")
    rng = np.random.default_rng(42)
    points, _ = DistributionGenerators().triangle_points(config.triangle, 1000, rng)
    assert points.shape == (1000, 3)


def test_sphere_directions_are_unit_length() -> None:
    rng = np.random.default_rng(42)
    points, _ = DistributionGenerators().sphere_directions(5000, rng)
    norms = np.linalg.norm(points, axis=1)
    assert np.max(np.abs(norms - 1.0)) < 1e-10

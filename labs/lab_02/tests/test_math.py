import math

from image_lab2.math.functions import analytic_integral_x_squared
from image_lab2.models.config import Interval


def test_analytic_integral_x_squared() -> None:
    value = analytic_integral_x_squared(Interval(2.0, 5.0))
    assert math.isclose(value, 39.0, rel_tol=1e-9)

from __future__ import annotations

from image_lab2.models.config import Interval


def evaluate_integrand(function_id: str, x):
    if function_id != "x_squared":
        raise ValueError("Unsupported integrand: {0}".format(function_id))
    return x * x


def analytic_integral_x_squared(interval: Interval) -> float:
    left = interval.left
    right = interval.right
    return (right**3 - left**3) / 3.0

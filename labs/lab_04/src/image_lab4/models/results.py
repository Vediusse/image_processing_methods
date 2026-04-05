from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvergencePoint:
    samples_per_pixel: int
    mse_to_reference: float
    mean_luminance: float


@dataclass(frozen=True)
class ReportMetrics:
    light_count: int
    triangle_count: int
    max_radiance: float
    mean_radiance: float
    convergence: list[ConvergencePoint]

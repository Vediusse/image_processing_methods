from __future__ import annotations

from dataclasses import dataclass

from .vector import Point3, Vector3


@dataclass(frozen=True)
class TriangleConfig:
    a: Point3
    b: Point3
    c: Point3


@dataclass(frozen=True)
class CircleConfig:
    center: Point3
    normal: Vector3
    radius: float


@dataclass(frozen=True)
class ExperimentConfig:
    sample_count: int
    seed: int
    triangle: TriangleConfig
    circle: CircleConfig
    cosine_normal: Vector3
    uniformity_rectangle_count: int = 4

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from .vector import Point3, Vector3


@dataclass(frozen=True)
class ColorRGB:
    r: float
    g: float
    b: float

    def __add__(self, other: "ColorRGB") -> "ColorRGB":
        return ColorRGB(self.r + other.r, self.g + other.g, self.b + other.b)

    def __mul__(self, value: Union[float, "ColorRGB"]) -> "ColorRGB":
        if isinstance(value, ColorRGB):
            return ColorRGB(self.r * value.r, self.g * value.g, self.b * value.b)
        return ColorRGB(self.r * value, self.g * value, self.b * value)

    def __rmul__(self, value: Union[float, "ColorRGB"]) -> "ColorRGB":
        return self.__mul__(value)

    def clamp(self, low: float = 0.0, high: float = 1.0) -> "ColorRGB":
        return ColorRGB(
            min(max(self.r, low), high),
            min(max(self.g, low), high),
            min(max(self.b, low), high),
        )

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.r, self.g, self.b)

    def to_vector(self) -> Vector3:
        return Vector3(self.r, self.g, self.b)

    @classmethod
    def black(cls) -> "ColorRGB":
        return cls(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Triangle:
    a: Point3
    b: Point3
    c: Point3


@dataclass(frozen=True)
class RadiationProfile:
    exponent: float = 1.0
    cutoff_degrees: float = 90.0


@dataclass(frozen=True)
class Light:
    name: str
    position: Point3
    axis_direction: Vector3
    intensity: ColorRGB
    profile: RadiationProfile = field(default_factory=RadiationProfile)


@dataclass(frozen=True)
class Material:
    color: ColorRGB
    diffuse_coefficient: float
    specular_coefficient: float
    shininess: float


@dataclass(frozen=True)
class Camera:
    position: Point3


@dataclass(frozen=True)
class LocalPoint:
    name: str
    u: float
    v: float


@dataclass(frozen=True)
class Scenario:
    triangle: Triangle
    lights: list[Light]
    material: Material
    observer: Camera
    local_points: list[LocalPoint]
    global_points: dict[str, Point3]

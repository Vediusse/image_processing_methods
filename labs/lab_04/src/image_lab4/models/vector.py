from __future__ import annotations

from dataclasses import dataclass
import math


EPSILON = 1e-8


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, value: float) -> "Vec3":
        if abs(value) < EPSILON:
            raise ValueError("Division by zero.")
        return Vec3(self.x / value, self.y / value, self.z / value)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        length = self.length()
        if length < EPSILON:
            raise ValueError("Cannot normalize zero vector.")
        return self / length

    def max_component(self) -> float:
        return max(self.x, self.y, self.z)

    def average(self) -> float:
        return (self.x + self.y + self.z) / 3.0

    def clamp(self, low: float = 0.0, high: float = 1.0) -> "Vec3":
        return Vec3(
            min(max(self.x, low), high),
            min(max(self.y, low), high),
            min(max(self.z, low), high),
        )

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def to_numpy(self):
        import numpy as np

        return np.array([self.x, self.y, self.z], dtype=float)

    @classmethod
    def zero(cls) -> "Vec3":
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def one(cls) -> "Vec3":
        return cls(1.0, 1.0, 1.0)


Point3 = Vec3
ColorRGB = Vec3

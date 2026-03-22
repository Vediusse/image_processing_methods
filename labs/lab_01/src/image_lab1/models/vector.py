from __future__ import annotations

from dataclasses import dataclass
import math


EPSILON = 1e-9


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector3":
        if abs(scalar) < EPSILON:
            raise ValueError("Division by zero scalar.")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vector3":
        norm = self.length()
        if norm < EPSILON:
            raise ValueError("Cannot normalize a zero-length vector.")
        return self / norm

    def distance_to(self, other: "Vector3") -> float:
        return (self - other).length()

    def clamp_min(self, value: float = 0.0) -> "Vector3":
        return Vector3(max(self.x, value), max(self.y, value), max(self.z, value))

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def almost_equal(self, other: "Vector3", tolerance: float = 1e-6) -> bool:
        return (
            abs(self.x - other.x) <= tolerance
            and abs(self.y - other.y) <= tolerance
            and abs(self.z - other.z) <= tolerance
        )


Point3 = Vector3

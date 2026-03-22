from __future__ import annotations

from image_lab3.models.vector import Vector3


def triangle_normal(a: Vector3, b: Vector3, c: Vector3) -> Vector3:
    return (b - a).cross(c - a).normalized()


def build_orthonormal_basis(normal: Vector3):
    n = normal.normalized()
    helper = Vector3(0.0, 0.0, 1.0)
    if abs(n.dot(helper)) > 0.95:
        helper = Vector3(0.0, 1.0, 0.0)
    tangent = n.cross(helper).normalized()
    bitangent = tangent.cross(n).normalized()
    return tangent, bitangent, n

from __future__ import annotations

import math
from typing import Tuple

from image_lab4.models.scene import Ray
from image_lab4.models.vector import EPSILON, Point3, Vec3


def orthonormal_basis(normal: Vec3) -> Tuple[Vec3, Vec3]:
    n = normal.normalized()
    if abs(n.x) > 0.1:
        tangent = Vec3(0.0, 1.0, 0.0).cross(n).normalized()
    else:
        tangent = Vec3(1.0, 0.0, 0.0).cross(n).normalized()
    bitangent = n.cross(tangent).normalized()
    return tangent, bitangent


def sample_cosine_weighted_hemisphere(normal: Vec3, u1: float, u2: float) -> Vec3:
    r = math.sqrt(u1)
    phi = 2.0 * math.pi * u2
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = math.sqrt(max(0.0, 1.0 - u1))
    tangent, bitangent = orthonormal_basis(normal)
    return (tangent * x + bitangent * y + normal.normalized() * z).normalized()


def reflect(direction: Vec3, normal: Vec3) -> Vec3:
    n = normal.normalized()
    return (direction - n * (2.0 * direction.dot(n))).normalized()


def triangle_area(a: Point3, b: Point3, c: Point3) -> float:
    return 0.5 * (b - a).cross(c - a).length()


def triangle_normal(a: Point3, b: Point3, c: Point3) -> Vec3:
    return (b - a).cross(c - a).normalized()


def sample_point_on_triangle(a: Point3, b: Point3, c: Point3, u1: float, u2: float) -> Point3:
    su1 = math.sqrt(u1)
    alpha = 1.0 - su1
    beta = su1 * (1.0 - u2)
    gamma = su1 * u2
    return a * alpha + b * beta + c * gamma


def moller_trumbore(ray: Ray, a: Point3, b: Point3, c: Point3):
    edge1 = b - a
    edge2 = c - a
    pvec = ray.direction.cross(edge2)
    det = edge1.dot(pvec)
    if abs(det) < EPSILON:
        return None
    inv_det = 1.0 / det
    tvec = ray.origin - a
    u = tvec.dot(pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return None
    qvec = tvec.cross(edge1)
    v = ray.direction.dot(qvec) * inv_det
    if v < 0.0 or u + v > 1.0:
        return None
    t = edge2.dot(qvec) * inv_det
    if t <= EPSILON:
        return None
    return t, u, v

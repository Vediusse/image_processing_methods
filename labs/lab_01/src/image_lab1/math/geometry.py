from __future__ import annotations

from image_lab1.models.scene import Triangle
from image_lab1.models.vector import Point3, Vector3


def validate_triangle(triangle: Triangle) -> None:
    edge_ab = triangle.b - triangle.a
    edge_ac = triangle.c - triangle.a
    if edge_ab.cross(edge_ac).length() <= 1e-9:
        raise ValueError("Triangle is degenerate: its vertices are collinear.")


def calculate_triangle_normal(triangle: Triangle) -> Vector3:
    validate_triangle(triangle)
    edge_ab = triangle.b - triangle.a
    edge_ac = triangle.c - triangle.a
    return edge_ab.cross(edge_ac).normalized()


def local_to_global(triangle: Triangle, u: float, v: float) -> Point3:
    validate_point_on_triangle(u, v)
    edge_ab = triangle.b - triangle.a
    edge_ac = triangle.c - triangle.a
    return triangle.a + edge_ab * u + edge_ac * v


def validate_point_on_triangle(u: float, v: float) -> None:
    if u < 0 or v < 0 or (u + v) > 1:
        raise ValueError(
            f"Local coordinates are outside the triangle: u={u}, v={v}, u+v={u + v}."
        )


def point_to_light_vector(point: Point3, light_position: Point3) -> Vector3:
    direction = light_position - point
    if direction.length() <= 1e-9:
        raise ValueError("Light source position coincides with the evaluated point.")
    return direction

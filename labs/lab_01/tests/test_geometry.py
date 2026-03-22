from image_lab1.math.geometry import calculate_triangle_normal, local_to_global
from image_lab1.models.scene import Triangle
from image_lab1.models.vector import Point3, Vector3


def test_local_to_global_conversion() -> None:
    triangle = Triangle(
        a=Point3(0.0, 0.0, 0.0),
        b=Point3(2.0, 0.0, 0.0),
        c=Point3(0.0, 2.0, 0.0),
    )
    point = local_to_global(triangle, 0.25, 0.5)
    assert point.almost_equal(Point3(0.5, 1.0, 0.0))


def test_triangle_normal() -> None:
    triangle = Triangle(
        a=Point3(0.0, 0.0, 0.0),
        b=Point3(1.0, 0.0, 0.0),
        c=Point3(0.0, 1.0, 0.0),
    )
    normal = calculate_triangle_normal(triangle)
    assert normal.almost_equal(Vector3(0.0, 0.0, 1.0))

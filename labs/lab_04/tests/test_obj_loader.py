from pathlib import Path

from image_lab4.io.obj_loader import load_obj_triangles
from image_lab4.models.scene import Transform
from image_lab4.models.vector import ColorRGB, Vec3


def test_obj_loader_reads_triangles() -> None:
    triangles = load_obj_triangles(
        path=Path("examples/pyramid.obj"),
        material_name="mirror",
        emission=ColorRGB.zero(),
        transform=Transform(translate=Vec3.zero(), scale=1.0),
    )
    assert len(triangles) == 6
    assert all(item.material_name == "mirror" for item in triangles)

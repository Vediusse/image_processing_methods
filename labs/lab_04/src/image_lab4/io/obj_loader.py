from __future__ import annotations

from pathlib import Path
from typing import List

from image_lab4.models.scene import Triangle, Transform
from image_lab4.models.vector import ColorRGB, Point3, Vec3


def load_obj_triangles(path: Path, material_name: str, emission: ColorRGB, transform: Transform) -> List[Triangle]:
    vertices = []
    triangles = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            _, x, y, z = line.split()[:4]
            point = Point3(float(x), float(y), float(z))
            point = point * transform.scale + transform.translate
            vertices.append(point)
        elif line.startswith("f "):
            face_tokens = line.split()[1:]
            indices = [int(token.split("/")[0]) - 1 for token in face_tokens]
            if len(indices) < 3:
                continue
            for offset in range(1, len(indices) - 1):
                a = vertices[indices[0]]
                b = vertices[indices[offset]]
                c = vertices[indices[offset + 1]]
                triangles.append(Triangle(a=a, b=b, c=c, material_name=material_name, emission=emission))
    return triangles

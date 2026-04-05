from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from image_lab4.models.scene import Camera, Material, ObjMesh, RenderSettings, SceneConfig, Transform, Triangle
from image_lab4.models.vector import ColorRGB, Point3, Vec3


def load_config(path: Union[str, Path]) -> SceneConfig:
    path = Path(path)
    return load_config_from_text(path.read_text(encoding="utf-8"), base_dir=path.parent)


def load_config_from_text(text: str, base_dir: Union[str, Path]) -> SceneConfig:
    data = json.loads(text)
    base_dir = Path(base_dir)
    materials = [
        Material(
            name=item["name"],
            diffuse=_color(item.get("diffuse", [0.0, 0.0, 0.0])),
            mirror=_color(item.get("mirror", [0.0, 0.0, 0.0])),
        )
        for item in data["materials"]
    ]
    triangles = [
        Triangle(
            a=_point(item["vertices"][0]),
            b=_point(item["vertices"][1]),
            c=_point(item["vertices"][2]),
            material_name=item["material"],
            emission=_color(item.get("emission", [0.0, 0.0, 0.0])),
        )
        for item in data.get("triangles", [])
    ]
    obj_meshes = [
        ObjMesh(
            path=str((base_dir / item["path"]).resolve()),
            material_name=item["material"],
            emission=_color(item.get("emission", [0.0, 0.0, 0.0])),
            transform=Transform(
                translate=_vec(item.get("transform", {}).get("translate", [0.0, 0.0, 0.0])),
                scale=float(item.get("transform", {}).get("scale", 1.0)),
            ),
        )
        for item in data.get("obj_meshes", [])
    ]
    return SceneConfig(
        camera=Camera(
            position=_point(data["camera"]["position"]),
            target=_point(data["camera"]["target"]),
            up=_vec(data["camera"]["up"]),
            fov_degrees=float(data["camera"]["fov_degrees"]),
        ),
        render=RenderSettings(
            width=int(data["render"]["width"]),
            height=int(data["render"]["height"]),
            samples_per_pixel=int(data["render"]["samples_per_pixel"]),
            max_depth=int(data["render"]["max_depth"]),
            min_depth=int(data["render"].get("min_depth", 2)),
            gamma=float(data["render"].get("gamma", 2.2)),
            normalization=str(data["render"].get("normalization", "max")),
            normalization_value=float(data["render"].get("normalization_value", 1.0)),
            seed=int(data["render"].get("seed", 0)),
            background=_color(data["render"].get("background", [0.0, 0.0, 0.0])),
        ),
        materials=materials,
        triangles=triangles,
        obj_meshes=obj_meshes,
    )


def save_config(path: Union[str, Path], config: SceneConfig) -> None:
    payload = {
        "camera": {
            "position": _serialize(config.camera.position),
            "target": _serialize(config.camera.target),
            "up": _serialize(config.camera.up),
            "fov_degrees": config.camera.fov_degrees,
        },
        "render": {
            "width": config.render.width,
            "height": config.render.height,
            "samples_per_pixel": config.render.samples_per_pixel,
            "max_depth": config.render.max_depth,
            "min_depth": config.render.min_depth,
            "gamma": config.render.gamma,
            "normalization": config.render.normalization,
            "normalization_value": config.render.normalization_value,
            "seed": config.render.seed,
            "background": list(config.render.background.to_tuple()),
        },
        "materials": [
            {
                "name": item.name,
                "diffuse": list(item.diffuse.to_tuple()),
                "mirror": list(item.mirror.to_tuple()),
            }
            for item in config.materials
        ],
        "triangles": [
            {
                "vertices": [list(item.a.to_tuple()), list(item.b.to_tuple()), list(item.c.to_tuple())],
                "material": item.material_name,
                "emission": list(item.emission.to_tuple()),
            }
            for item in config.triangles
        ],
        "obj_meshes": [
            {
                "path": item.path,
                "material": item.material_name,
                "emission": list(item.emission.to_tuple()),
                "transform": {
                    "translate": list(item.transform.translate.to_tuple()),
                    "scale": item.transform.scale,
                },
            }
            for item in config.obj_meshes
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _point(values):
    return Point3(float(values[0]), float(values[1]), float(values[2]))


def _vec(values):
    return Vec3(float(values[0]), float(values[1]), float(values[2]))


def _color(values):
    return ColorRGB(float(values[0]), float(values[1]), float(values[2]))


def _serialize(vec: Vec3):
    return [vec.x, vec.y, vec.z]

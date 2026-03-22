from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from image_lab1.models.scene import (
    Camera,
    ColorRGB,
    Light,
    LocalPoint,
    Material,
    RadiationProfile,
    Scenario,
    Triangle,
)
from image_lab1.models.vector import Point3, Vector3


def _point3(data: dict) -> Point3:
    return Point3(float(data["x"]), float(data["y"]), float(data["z"]))


def _vector3(data: dict) -> Vector3:
    return Vector3(float(data["x"]), float(data["y"]), float(data["z"]))


def _color(data: dict) -> ColorRGB:
    return ColorRGB(float(data["r"]), float(data["g"]), float(data["b"]))


def load_scenario(path: Union[str, Path]) -> Scenario:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    triangle = Triangle(
        a=_point3(data["triangle"]["a"]),
        b=_point3(data["triangle"]["b"]),
        c=_point3(data["triangle"]["c"]),
    )
    lights = [
        Light(
            name=item["name"],
            position=_point3(item["position"]),
            axis_direction=_vector3(item["axis_direction"]),
            intensity=_color(item["intensity"]),
            profile=RadiationProfile(
                exponent=float(item["profile"]["exponent"]),
                cutoff_degrees=float(item["profile"]["cutoff_degrees"]),
            ),
        )
        for item in data["lights"]
    ]
    material = Material(
        color=_color(data["material"]["color"]),
        diffuse_coefficient=float(data["material"]["diffuse_coefficient"]),
        specular_coefficient=float(data["material"]["specular_coefficient"]),
        shininess=float(data["material"]["shininess"]),
    )
    observer = Camera(position=_point3(data["observer"]["position"]))
    local_points = [
        LocalPoint(name=item["name"], u=float(item["u"]), v=float(item["v"]))
        for item in data["points"]["local"]
    ]
    global_points = {
        key: _point3(value) for key, value in data["points"]["global"].items()
    }
    return Scenario(
        triangle=triangle,
        lights=lights,
        material=material,
        observer=observer,
        local_points=local_points,
        global_points=global_points,
    )


def save_scenario(path: Union[str, Path], scenario: Scenario) -> None:
    content = {
        "triangle": {
            "a": _serialize_point(scenario.triangle.a),
            "b": _serialize_point(scenario.triangle.b),
            "c": _serialize_point(scenario.triangle.c),
        },
        "lights": [
            {
                "name": light.name,
                "position": _serialize_point(light.position),
                "axis_direction": _serialize_point(light.axis_direction),
                "intensity": _serialize_color(light.intensity),
                "profile": {
                    "exponent": light.profile.exponent,
                    "cutoff_degrees": light.profile.cutoff_degrees,
                },
            }
            for light in scenario.lights
        ],
        "material": {
            "color": _serialize_color(scenario.material.color),
            "diffuse_coefficient": scenario.material.diffuse_coefficient,
            "specular_coefficient": scenario.material.specular_coefficient,
            "shininess": scenario.material.shininess,
        },
        "observer": {"position": _serialize_point(scenario.observer.position)},
        "points": {
            "local": [
                {"name": point.name, "u": point.u, "v": point.v}
                for point in scenario.local_points
            ],
            "global": {
                name: _serialize_point(point)
                for name, point in scenario.global_points.items()
            },
        },
    }
    Path(path).write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def _serialize_point(point: Union[Point3, Vector3]) -> dict[str, float]:
    return {"x": point.x, "y": point.y, "z": point.z}


def _serialize_color(color: ColorRGB) -> dict[str, float]:
    return {"r": color.r, "g": color.g, "b": color.b}

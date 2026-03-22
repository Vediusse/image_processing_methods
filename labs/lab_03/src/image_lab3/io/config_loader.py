from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from image_lab3.models.config import CircleConfig, ExperimentConfig, TriangleConfig
from image_lab3.models.vector import Point3, Vector3


def _point(data):
    return Point3(float(data["x"]), float(data["y"]), float(data["z"]))


def _vector(data):
    return Vector3(float(data["x"]), float(data["y"]), float(data["z"]))


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        sample_count=int(data["sample_count"]),
        seed=int(data["seed"]),
        triangle=TriangleConfig(
            a=_point(data["triangle"]["a"]),
            b=_point(data["triangle"]["b"]),
            c=_point(data["triangle"]["c"]),
        ),
        circle=CircleConfig(
            center=_point(data["circle"]["center"]),
            normal=_vector(data["circle"]["normal"]),
            radius=float(data["circle"]["radius"]),
        ),
        cosine_normal=_vector(data["cosine_normal"]),
    )


def save_config(path: Union[str, Path], config: ExperimentConfig) -> None:
    content = {
        "sample_count": config.sample_count,
        "seed": config.seed,
        "triangle": {
            "a": _serialize_vec(config.triangle.a),
            "b": _serialize_vec(config.triangle.b),
            "c": _serialize_vec(config.triangle.c),
        },
        "circle": {
            "center": _serialize_vec(config.circle.center),
            "normal": _serialize_vec(config.circle.normal),
            "radius": config.circle.radius,
        },
        "cosine_normal": _serialize_vec(config.cosine_normal),
    }
    Path(path).write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def _serialize_vec(vec):
    return {"x": vec.x, "y": vec.y, "z": vec.z}

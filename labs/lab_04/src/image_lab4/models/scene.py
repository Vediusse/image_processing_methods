from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .vector import ColorRGB, Point3, Vec3


@dataclass(frozen=True)
class Material:
    name: str
    diffuse: ColorRGB
    mirror: ColorRGB


@dataclass(frozen=True)
class Triangle:
    a: Point3
    b: Point3
    c: Point3
    material_name: str
    emission: ColorRGB = field(default_factory=ColorRGB.zero)


@dataclass(frozen=True)
class Transform:
    translate: Vec3 = field(default_factory=Vec3.zero)
    scale: float = 1.0


@dataclass(frozen=True)
class ObjMesh:
    path: str
    material_name: str
    emission: ColorRGB = field(default_factory=ColorRGB.zero)
    transform: Transform = field(default_factory=Transform)


@dataclass(frozen=True)
class Camera:
    position: Point3
    target: Point3
    up: Vec3
    fov_degrees: float


@dataclass(frozen=True)
class RenderSettings:
    width: int
    height: int
    samples_per_pixel: int
    max_depth: int
    min_depth: int
    gamma: float
    normalization: str
    normalization_value: float
    seed: int
    background: ColorRGB = field(default_factory=ColorRGB.zero)


@dataclass(frozen=True)
class SceneConfig:
    camera: Camera
    render: RenderSettings
    materials: List[Material]
    triangles: List[Triangle]
    obj_meshes: List[ObjMesh]


@dataclass(frozen=True)
class ResolvedTriangle:
    a: Point3
    b: Point3
    c: Point3
    material: Material
    emission: ColorRGB
    area: float
    normal: Vec3


@dataclass(frozen=True)
class Scene:
    camera: Camera
    render: RenderSettings
    triangles: List[ResolvedTriangle]
    lights: List[ResolvedTriangle]
    camera_forward: Vec3
    camera_right: Vec3
    camera_up: Vec3
    tan_half_fov: float
    triangle_a: "np.ndarray"
    triangle_edge1: "np.ndarray"
    triangle_edge2: "np.ndarray"
    triangle_normals: "np.ndarray"
    light_indices: "np.ndarray"
    light_probabilities: "np.ndarray"


@dataclass(frozen=True)
class Ray:
    origin: Point3
    direction: Vec3


@dataclass(frozen=True)
class HitRecord:
    distance: float
    position: Point3
    normal: Vec3
    triangle: ResolvedTriangle
    triangle_index: int


@dataclass(frozen=True)
class RenderArtifact:
    radiance: "np.ndarray"
    display: "np.ndarray"
    summary: str
    scene: Scene
    config_path: Optional[Path] = None

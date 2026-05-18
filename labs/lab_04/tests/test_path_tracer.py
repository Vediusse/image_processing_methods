from pathlib import Path

import numpy as np

from image_lab4.io.config_loader import load_config
from image_lab4.models.scene import Camera, Material, PointLight, RenderSettings, SceneConfig, Triangle
from image_lab4.models.vector import ColorRGB, Point3, Vec3
from image_lab4.services.path_tracer import PathTracer


def test_default_scene_traces_central_ray() -> None:
    config = load_config(Path("examples/default_scene.json"))
    tracer = PathTracer()
    scene = tracer._build_scene(config)
    rng = np.random.default_rng(1)
    ray = tracer._generate_camera_ray(scene, scene.render.width // 2, scene.render.height // 2, rng)
    radiance = tracer._trace_path(scene, ray, rng)
    assert radiance.shape == (3,)
    assert np.isfinite(radiance).all()
    assert len(scene.lights) >= 1
    assert len(scene.triangles) >= 1


def test_energy_conservation_violation_raises() -> None:
    config = load_config(Path("examples/default_scene.json"))
    broken_material = config.materials[0].__class__(
        name="broken",
        diffuse=config.materials[0].diffuse.__class__(0.8, 0.8, 0.8),
        mirror=config.materials[0].mirror.__class__(0.4, 0.4, 0.4),
    )
    broken_config = config.__class__(
        camera=config.camera,
        render=config.render,
        materials=[broken_material] + config.materials[1:],
        triangles=config.triangles,
        obj_meshes=config.obj_meshes,
    )
    try:
        PathTracer().render(broken_config)
    except ValueError as error:
        assert "energy conservation" in str(error)
    else:
        raise AssertionError("Expected energy conservation validation error.")


def test_area_and_point_lights_use_common_flux_units_for_probability() -> None:
    tracer = PathTracer()
    materials = [Material(name="white", diffuse=ColorRGB(0.8, 0.8, 0.8), mirror=ColorRGB.zero())]
    triangles = [
        Triangle(
            a=Point3(0.0, 0.0, 0.0),
            b=Point3(2.0, 0.0, 0.0),
            c=Point3(0.0, 2.0, 0.0),
            material_name="white",
            emission=ColorRGB(1.0, 1.0, 1.0),
        )
    ]
    point_lights = [
        PointLight(
            position=Point3(0.0, 1.0, 1.0),
            intensity=ColorRGB(0.5, 0.5, 0.5),
        )
    ]
    config = SceneConfig(
        camera=Camera(
            position=Point3(0.0, 0.0, 3.0),
            target=Point3(0.0, 0.0, 0.0),
            up=Vec3(0.0, 1.0, 0.0),
            fov_degrees=45.0,
        ),
        render=RenderSettings(
            width=64,
            height=64,
            samples_per_pixel=1,
            max_depth=2,
            min_depth=1,
            gamma=2.2,
            normalization="max",
            normalization_value=1.0,
            seed=1,
            background=ColorRGB.zero(),
        ),
        materials=materials,
        triangles=triangles,
        obj_meshes=[],
        point_lights=point_lights,
    )
    scene = tracer._build_scene(config, strict_resolution=False)
    assert len(scene.light_probabilities) == 2
    assert np.allclose(scene.light_probabilities, np.array([0.5, 0.5]), atol=1e-6)

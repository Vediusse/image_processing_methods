from pathlib import Path

import numpy as np

from image_lab4.io.config_loader import load_config
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

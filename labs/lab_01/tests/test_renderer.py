from image_lab1.io.scenario_loader import load_scenario
from image_lab1.services.observer_renderer import ObserverRenderer


def test_observer_renderer_produces_visible_surface() -> None:
    scenario = load_scenario("examples/demo_scenario.json")
    result = ObserverRenderer(width=240, height=240, fov_degrees=34.0).render(scenario)
    assert result.hit_pixels > 0
    assert result.image.shape == (240, 240, 3)

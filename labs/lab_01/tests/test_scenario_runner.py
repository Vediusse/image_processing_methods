from image_lab1.math.geometry import calculate_triangle_normal, local_to_global
from image_lab1.models.scene import Camera, Scenario
from image_lab1.io.scenario_loader import load_scenario
from image_lab1.services.scenario_runner import ScenarioRunner


def test_demo_scenario_runs() -> None:
    scenario = load_scenario("examples/demo_scenario.json")
    result = ScenarioRunner().run(scenario)
    assert len(result.local_evaluations) == 3
    assert len(result.global_evaluations) == 3
    assert all(item.total_brightness.r >= 0 for item in result.local_evaluations)


def test_backfacing_observer_keeps_irradiance_but_removes_brightness() -> None:
    scenario = load_scenario("examples/demo_scenario.json")
    normal = calculate_triangle_normal(scenario.triangle)
    reference_point = local_to_global(scenario.triangle, 0.2, 0.2)
    observer_below_surface = reference_point - normal * 2.5
    dark_side_scenario = Scenario(
        triangle=scenario.triangle,
        lights=scenario.lights,
        material=scenario.material,
        observer=Camera(position=observer_below_surface),
        local_points=scenario.local_points,
        global_points=scenario.global_points,
    )

    result = ScenarioRunner().run(dark_side_scenario)
    point = result.local_evaluations[0]

    assert point.total_irradiance.r > 0.0
    assert point.total_irradiance.g > 0.0
    assert point.total_irradiance.b > 0.0
    assert point.total_brightness.r == 0.0
    assert point.total_brightness.g == 0.0
    assert point.total_brightness.b == 0.0

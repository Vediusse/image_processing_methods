from image_lab1.io.scenario_loader import load_scenario
from image_lab1.services.scenario_runner import ScenarioRunner


def test_demo_scenario_runs() -> None:
    scenario = load_scenario("examples/demo_scenario.json")
    result = ScenarioRunner().run(scenario)
    assert len(result.local_evaluations) == 3
    assert len(result.global_evaluations) == 3
    assert all(item.total_brightness.r >= 0 for item in result.local_evaluations)

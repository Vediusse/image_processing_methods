from image_lab2.io.config_loader import load_config
from image_lab2.services.experiment_runner import ExperimentRunner


def test_runner_produces_all_method_series() -> None:
    config = load_config("examples/default_config.json")
    result = ExperimentRunner().run(config)
    assert result.true_value > 0
    assert "simple_mc" in result.by_method
    assert "mis_balance" in result.by_method
    assert "mis_power" in result.by_method

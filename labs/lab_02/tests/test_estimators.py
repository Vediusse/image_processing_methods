from image_lab2.io.config_loader import load_config
from image_lab2.services.estimators import MonteCarloEstimators


def test_simple_mc_converges_near_true_value_for_large_n() -> None:
    config = load_config("examples/default_config.json")
    estimators = MonteCarloEstimators(config)
    estimate = estimators.simple_mc(100000, 12345, 39.0)
    assert abs(estimate.estimate - 39.0) < 0.4


def test_importance_sampling_runs_for_power_three() -> None:
    config = load_config("examples/default_config.json")
    estimators = MonteCarloEstimators(config)
    estimate = estimators.importance_sampling(10000, 3, 111, 39.0)
    assert estimate.estimate > 0

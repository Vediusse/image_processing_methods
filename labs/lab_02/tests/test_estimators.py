from statistics import mean

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


def test_adaptive_stratification_beats_simple_mc_on_average() -> None:
    config = load_config("examples/default_config.json")
    estimators = MonteCarloEstimators(config)

    simple_errors = [estimators.simple_mc(200, 5000 + index, 39.0).absolute_error for index in range(48)]
    stratified_errors = [estimators.stratified(200, 0.5, 5000 + index, 39.0).absolute_error for index in range(48)]

    assert mean(stratified_errors) < mean(simple_errors) * 0.3


def test_smaller_stratification_step_reduces_average_error() -> None:
    config = load_config("examples/default_config.json")
    estimators = MonteCarloEstimators(config)

    coarse_errors = [estimators.stratified(200, 1.0, 7000 + index, 39.0).absolute_error for index in range(48)]
    fine_errors = [estimators.stratified(200, 0.5, 7000 + index, 39.0).absolute_error for index in range(48)]

    assert mean(fine_errors) < mean(coarse_errors) * 0.6

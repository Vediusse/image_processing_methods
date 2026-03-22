import math

from image_lab3.io.config_loader import load_config
from image_lab3.services.experiment_runner import ExperimentRunner


def test_runner_returns_all_distributions() -> None:
    config = load_config("examples/default_config.json")
    result = ExperimentRunner().run(config)
    assert set(result.distributions.keys()) == {"triangle", "circle", "sphere", "cosine"}
    assert result.sample_count == 100000


def test_no_distribution_metric_is_nan() -> None:
    config = load_config("examples/default_config.json")
    result = ExperimentRunner().run(config)
    for distribution in result.distributions.values():
        metrics = distribution.metrics
        assert math.isfinite(metrics.inside_ratio)
        assert math.isfinite(metrics.norm_error)
        assert math.isfinite(metrics.centroid_or_mean_error)
        assert math.isfinite(metrics.uniformity_score)

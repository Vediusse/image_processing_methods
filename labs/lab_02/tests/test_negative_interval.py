import math

from image_lab2.models.config import ExperimentConfig, Interval
from image_lab2.services.experiment_runner import ExperimentRunner


def test_negative_interval_does_not_produce_nan() -> None:
    config = ExperimentConfig(
        integrand="x_squared",
        interval=Interval(-5.0, -2.0),
        sample_sizes=[1000],
        stratification_steps=[1.0, 0.5],
        importance_sampling_powers=[1, 2, 3],
        mis_powers=[1, 3],
        russian_roulette_thresholds=[0.5, 0.75, 0.95],
        seed=12345,
    )
    result = ExperimentRunner().run(config)
    for series in result.by_method.values():
        for estimate in series.estimates:
            assert math.isfinite(estimate.estimate)
            assert math.isfinite(estimate.absolute_error)
            assert math.isfinite(estimate.estimated_error)

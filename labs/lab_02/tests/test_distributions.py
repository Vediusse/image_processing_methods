import numpy as np

from image_lab2.math.distributions import PowerPdf
from image_lab2.models.config import Interval


def test_power_pdf_samples_are_within_interval() -> None:
    distribution = PowerPdf(Interval(2.0, 5.0), 2)
    rng = np.random.default_rng(42)
    samples = distribution.sample(rng, 1000)
    assert samples.min() >= 2.0
    assert samples.max() <= 5.0

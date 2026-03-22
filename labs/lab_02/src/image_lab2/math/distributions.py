from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from image_lab2.models.config import Interval


@dataclass(frozen=True)
class PowerPdf:
    interval: Interval
    power: int

    def uses_shifted_density(self) -> bool:
        return self.interval.left < 0.0

    def pdf(self, x):
        values = np.asarray(x, dtype=np.float64)
        if self.uses_shifted_density():
            shifted = np.maximum(values - self.interval.left, 0.0)
            normalization = (self.interval.width ** (self.power + 1)) / (self.power + 1)
            return np.power(shifted, self.power) / normalization

        normalization = (self.interval.right ** (self.power + 1) - self.interval.left ** (self.power + 1)) / (
            self.power + 1
        )
        return np.power(values, self.power) / normalization

    def sample(self, rng: np.random.Generator, size: int):
        u = rng.random(size)
        if self.uses_shifted_density():
            shifted = self.interval.width * np.power(u, 1.0 / (self.power + 1))
            return self.interval.left + shifted

        left_term = self.interval.left ** (self.power + 1)
        right_term = self.interval.right ** (self.power + 1)
        values = left_term + u * (right_term - left_term)
        return np.power(values, 1.0 / (self.power + 1))

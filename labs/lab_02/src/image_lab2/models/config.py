from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Interval:
    left: float
    right: float

    @property
    def width(self) -> float:
        return self.right - self.left


@dataclass(frozen=True)
class ExperimentConfig:
    integrand: str
    interval: Interval
    sample_sizes: List[int]
    stratification_steps: List[float]
    importance_sampling_powers: List[int]
    mis_powers: List[int]
    russian_roulette_thresholds: List[float]
    seed: int

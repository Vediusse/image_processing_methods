from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class DistributionMetrics:
    title: str
    inside_ratio: float
    norm_error: float
    centroid_or_mean_error: float
    uniformity_score: float
    note: str


@dataclass(frozen=True)
class DistributionResult:
    key: str
    title: str
    points: np.ndarray
    metrics: DistributionMetrics
    auxiliary: Dict[str, np.ndarray]


@dataclass(frozen=True)
class ExperimentResult:
    sample_count: int
    distributions: Dict[str, DistributionResult]

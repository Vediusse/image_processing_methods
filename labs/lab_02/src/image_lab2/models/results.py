from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MethodEstimate:
    method_key: str
    display_name: str
    sample_size: int
    estimate: float
    true_value: float
    absolute_error: float
    relative_error: float
    estimated_error: float


@dataclass(frozen=True)
class MethodSeries:
    method_key: str
    display_name: str
    estimates: List[MethodEstimate]


@dataclass(frozen=True)
class ExperimentResult:
    true_value: float
    estimated_error_formula: str
    by_method: Dict[str, MethodSeries]

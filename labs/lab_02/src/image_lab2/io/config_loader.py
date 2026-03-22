from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from image_lab2.models.config import ExperimentConfig, Interval


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        integrand=data["integrand"],
        interval=Interval(
            left=float(data["interval"]["left"]),
            right=float(data["interval"]["right"]),
        ),
        sample_sizes=[int(value) for value in data["sample_sizes"]],
        stratification_steps=[float(value) for value in data["stratification_steps"]],
        importance_sampling_powers=[int(value) for value in data["importance_sampling_powers"]],
        mis_powers=[int(value) for value in data["mis_powers"]],
        russian_roulette_thresholds=[float(value) for value in data["russian_roulette_thresholds"]],
        seed=int(data["seed"]),
    )


def save_config(path: Union[str, Path], config: ExperimentConfig) -> None:
    content = {
        "integrand": config.integrand,
        "interval": {
            "left": config.interval.left,
            "right": config.interval.right,
        },
        "sample_sizes": config.sample_sizes,
        "stratification_steps": config.stratification_steps,
        "importance_sampling_powers": config.importance_sampling_powers,
        "mis_powers": config.mis_powers,
        "russian_roulette_thresholds": config.russian_roulette_thresholds,
        "seed": config.seed,
    }
    Path(path).write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")

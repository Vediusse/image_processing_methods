from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from image_lab3.models.results import ExperimentResult


def export_json(result: ExperimentResult, destination: Union[str, Path]) -> None:
    payload = {
        "sample_count": result.sample_count,
        "distributions": {
            key: {
                "title": distribution.title,
                "metrics": {
                    "inside_ratio": distribution.metrics.inside_ratio,
                    "norm_error": distribution.metrics.norm_error,
                    "centroid_or_mean_error": distribution.metrics.centroid_or_mean_error,
                    "uniformity_score": distribution.metrics.uniformity_score,
                    "note": distribution.metrics.note,
                },
            }
            for key, distribution in result.distributions.items()
        },
    }
    Path(destination).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_markdown(result: ExperimentResult, destination: Union[str, Path]) -> None:
    lines = [
        "# ЛР 3 — Результаты проверки распределений",
        "",
        "| Распределение | Inside ratio | Norm error | Mean/Centroid error | Uniformity score |",
        "|---|---:|---:|---:|---:|",
    ]
    for distribution in result.distributions.values():
        metrics = distribution.metrics
        lines.append(
            "| {title} | {inside:.6f} | {norm:.6f} | {mean_err:.6f} | {uniformity:.6f} |".format(
                title=distribution.title,
                inside=metrics.inside_ratio,
                norm=metrics.norm_error,
                mean_err=metrics.centroid_or_mean_error,
                uniformity=metrics.uniformity_score,
            )
        )
    Path(destination).write_text("\n".join(lines), encoding="utf-8")

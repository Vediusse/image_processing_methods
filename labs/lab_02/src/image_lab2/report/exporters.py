from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Union

from image_lab2.models.results import ExperimentResult


def export_json(result: ExperimentResult, destination: Union[str, Path]) -> None:
    path = Path(destination)
    payload = {
        "true_value": result.true_value,
        "estimated_error_formula": result.estimated_error_formula,
        "methods": {
            key: [
                {
                    "sample_size": estimate.sample_size,
                    "estimate": estimate.estimate,
                    "absolute_error": estimate.absolute_error,
                    "relative_error": estimate.relative_error,
                    "estimated_error": estimate.estimated_error,
                }
                for estimate in series.estimates
            ]
            for key, series in result.by_method.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_csv(result: ExperimentResult, destination: Union[str, Path]) -> None:
    path = Path(destination)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "method_key",
                "display_name",
                "sample_size",
                "true_value",
                "estimate",
                "absolute_error",
                "relative_error",
                "estimated_error",
            ]
        )
        for series in result.by_method.values():
            for estimate in series.estimates:
                writer.writerow(
                    [
                        estimate.method_key,
                        estimate.display_name,
                        estimate.sample_size,
                        estimate.true_value,
                        estimate.estimate,
                        estimate.absolute_error,
                        estimate.relative_error,
                        estimate.estimated_error,
                    ]
                )


def export_markdown(result: ExperimentResult, destination: Union[str, Path]) -> None:
    lines = [
        "# ЛР 2 — Результаты интегрирования",
        "",
        "Истинное значение интеграла: `{0:.8f}`".format(result.true_value),
        "",
        "| Метод | N | Оценка | Абс. ошибка | Отн. ошибка | Оценка погрешности |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for series in result.by_method.values():
        for estimate in series.estimates:
            lines.append(
                "| {name} | {n} | {value:.8f} | {abs_err:.8f} | {rel_err:.8f} | {stderr:.8f} |".format(
                    name=estimate.display_name,
                    n=estimate.sample_size,
                    value=estimate.estimate,
                    abs_err=estimate.absolute_error,
                    rel_err=estimate.relative_error,
                    stderr=estimate.estimated_error,
                )
            )
    Path(destination).write_text("\n".join(lines), encoding="utf-8")

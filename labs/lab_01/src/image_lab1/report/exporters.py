from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Union

from image_lab1.models.results import BrightnessResult, PointEvaluation


def export_json(result: BrightnessResult, destination: Union[str, Path]) -> None:
    path = Path(destination)
    payload = {
        "local_evaluations": [_evaluation_to_dict(item) for item in result.local_evaluations],
        "global_evaluations": [_evaluation_to_dict(item) for item in result.global_evaluations],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_csv(result: BrightnessResult, destination: Union[str, Path]) -> None:
    path = Path(destination)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "scope",
                "point",
                "u",
                "v",
                "x",
                "y",
                "z",
                "irradiance_r",
                "irradiance_g",
                "irradiance_b",
                "brightness_r",
                "brightness_g",
                "brightness_b",
            ]
        )
        for scope, evaluations in (
            ("local", result.local_evaluations),
            ("global", result.global_evaluations),
        ):
            for item in evaluations:
                writer.writerow(
                    [
                        scope,
                        item.point_name,
                        item.local_coordinates[0],
                        item.local_coordinates[1],
                        item.global_point.x,
                        item.global_point.y,
                        item.global_point.z,
                        item.total_irradiance.r,
                        item.total_irradiance.g,
                        item.total_irradiance.b,
                        item.total_brightness.r,
                        item.total_brightness.g,
                        item.total_brightness.b,
                    ]
                )


def export_markdown(result: BrightnessResult, destination: Union[str, Path]) -> None:
    path = Path(destination)
    lines = [
        "# Lab 1 Results",
        "",
        "## Local Coordinates Evaluation",
        "",
        _table_for_evaluations(result.local_evaluations),
        "",
        "## Global Coordinates Evaluation",
        "",
        _table_for_evaluations(result.global_evaluations),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _table_for_evaluations(evaluations: list[PointEvaluation]) -> str:
    header = (
        "| Point | Local (u, v) | Global (x, y, z) | Irradiance RGB | Brightness RGB |\n"
        "|---|---|---|---|---|"
    )
    rows = []
    for item in evaluations:
        rows.append(
            "| {name} | ({u:.3f}, {v:.3f}) | ({x:.3f}, {y:.3f}, {z:.3f}) | "
            "({er:.5f}, {eg:.5f}, {eb:.5f}) | ({br:.5f}, {bg:.5f}, {bb:.5f}) |".format(
                name=item.point_name,
                u=item.local_coordinates[0],
                v=item.local_coordinates[1],
                x=item.global_point.x,
                y=item.global_point.y,
                z=item.global_point.z,
                er=item.total_irradiance.r,
                eg=item.total_irradiance.g,
                eb=item.total_irradiance.b,
                br=item.total_brightness.r,
                bg=item.total_brightness.g,
                bb=item.total_brightness.b,
            )
        )
    return "\n".join([header, *rows])


def _evaluation_to_dict(item: PointEvaluation) -> dict:
    return {
        "point_name": item.point_name,
        "local_coordinates": {"u": item.local_coordinates[0], "v": item.local_coordinates[1]},
        "global_point": {
            "x": item.global_point.x,
            "y": item.global_point.y,
            "z": item.global_point.z,
        },
        "normal": {"x": item.normal.x, "y": item.normal.y, "z": item.normal.z},
        "observer_direction": {
            "x": item.observer_direction.x,
            "y": item.observer_direction.y,
            "z": item.observer_direction.z,
        },
        "total_irradiance": {
            "r": item.total_irradiance.r,
            "g": item.total_irradiance.g,
            "b": item.total_irradiance.b,
        },
        "total_brightness": {
            "r": item.total_brightness.r,
            "g": item.total_brightness.g,
            "b": item.total_brightness.b,
        },
        "contributions": [
            {
                "light_name": contribution.light_name,
                "distance": contribution.distance,
                "theta_degrees": contribution.theta_degrees,
                "alpha_degrees": contribution.alpha_degrees,
                "angular_profile": contribution.angular_profile,
                "colored_intensity": {
                    "r": contribution.colored_intensity.r,
                    "g": contribution.colored_intensity.g,
                    "b": contribution.colored_intensity.b,
                },
                "colored_irradiance": {
                    "r": contribution.colored_irradiance.r,
                    "g": contribution.colored_irradiance.g,
                    "b": contribution.colored_irradiance.b,
                },
                "brdf_value": contribution.brdf_value,
                "brightness": {
                    "r": contribution.brightness.r,
                    "g": contribution.brightness.g,
                    "b": contribution.brightness.b,
                },
            }
            for contribution in item.contributions
        ],
    }

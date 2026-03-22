from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from image_lab1.io.scenario_loader import load_scenario
from image_lab1.report.exporters import export_csv, export_json, export_markdown
from image_lab1.services.scenario_runner import ScenarioRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab 1 triangle brightness scenario.")
    parser.add_argument(
        "--scenario",
        default="examples/demo_scenario.json",
        help="Path to input scenario JSON.",
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Optional directory for JSON, CSV, and Markdown exports.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    console = Console()
    scenario = load_scenario(args.scenario)
    result = ScenarioRunner().run(scenario)

    console.print("[bold cyan]Lab 1: Triangle Illumination and Brightness[/bold cyan]")
    console.print()
    _render_scope(console, "Local Coordinates", result.local_evaluations)
    console.print()
    _render_scope(console, "Global Coordinates", result.global_evaluations)

    if args.export_dir:
        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        export_json(result, export_dir / "lab1_results.json")
        export_csv(result, export_dir / "lab1_results.csv")
        export_markdown(result, export_dir / "lab1_results.md")
        console.print(f"\n[green]Exports saved to[/green] {export_dir}")


def _render_scope(console: Console, title: str, evaluations) -> None:
    table = Table(title=title)
    for column in ("Point", "Local", "Global", "Irradiance RGB", "Brightness RGB"):
        table.add_column(column)
    for item in evaluations:
        table.add_row(
            item.point_name,
            f"({item.local_coordinates[0]:.3f}, {item.local_coordinates[1]:.3f})",
            f"({item.global_point.x:.3f}, {item.global_point.y:.3f}, {item.global_point.z:.3f})",
            f"({item.total_irradiance.r:.5f}, {item.total_irradiance.g:.5f}, {item.total_irradiance.b:.5f})",
            f"({item.total_brightness.r:.5f}, {item.total_brightness.g:.5f}, {item.total_brightness.b:.5f})",
        )
    console.print(table)

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from image_lab3.io.config_loader import load_config
from image_lab3.report.exporters import export_json, export_markdown
from image_lab3.services.experiment_runner import ExperimentRunner


def build_parser():
    parser = argparse.ArgumentParser(description="Run random distributions lab 3.")
    parser.add_argument("--config", default="examples/default_config.json", help="Path to config JSON.")
    parser.add_argument("--export-dir", default=None, help="Directory for export files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = ExperimentRunner().run(load_config(args.config))
    console = Console()
    console.print("[bold cyan]ЛР 3 — Формирование распределений случайных величин[/bold cyan]\n")
    table = Table(title="Метрики корректности")
    for column in ("Распределение", "Inside ratio", "Norm error", "Mean error", "Uniformity score"):
        table.add_column(column)
    for distribution in result.distributions.values():
        metrics = distribution.metrics
        table.add_row(
            distribution.title,
            "{0:.6f}".format(metrics.inside_ratio),
            "{0:.6e}".format(metrics.norm_error),
            "{0:.6e}".format(metrics.centroid_or_mean_error),
            "{0:.6f}".format(metrics.uniformity_score),
        )
    console.print(table)
    if args.export_dir:
        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        export_json(result, export_dir / "lab3_results.json")
        export_markdown(result, export_dir / "lab3_results.md")
        console.print("\n[green]Результаты экспортированы в[/green] {0}".format(export_dir))


if __name__ == "__main__":
    main()

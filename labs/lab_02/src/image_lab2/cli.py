from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from image_lab2.io.config_loader import load_config
from image_lab2.report.exporters import export_csv, export_json, export_markdown
from image_lab2.services.experiment_runner import ExperimentRunner


def build_parser():
    parser = argparse.ArgumentParser(description="Run Monte Carlo integration lab 2.")
    parser.add_argument("--config", default="examples/default_config.json", help="Path to config JSON.")
    parser.add_argument("--export-dir", default=None, help="Directory for export files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    result = ExperimentRunner().run(config)
    console = Console()
    console.print("[bold cyan]ЛР 2 — Интегрирование методом Монте-Карло[/bold cyan]")
    console.print("Истинный интеграл: [bold]{0:.8f}[/bold]\n".format(result.true_value))

    table = Table(title="Сводная таблица результатов")
    for column in ("Метод", "N", "Оценка", "Абс. ошибка", "Отн. ошибка", "Оценка погрешности"):
        table.add_column(column)
    for series in result.by_method.values():
        for estimate in series.estimates:
            table.add_row(
                estimate.display_name,
                str(estimate.sample_size),
                "{0:.8f}".format(estimate.estimate),
                "{0:.8f}".format(estimate.absolute_error),
                "{0:.8f}".format(estimate.relative_error),
                "{0:.8f}".format(estimate.estimated_error),
            )
    console.print(table)

    if args.export_dir:
        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        export_json(result, export_dir / "lab2_results.json")
        export_csv(result, export_dir / "lab2_results.csv")
        export_markdown(result, export_dir / "lab2_results.md")
        console.print("\n[green]Результаты экспортированы в[/green] {0}".format(export_dir))


if __name__ == "__main__":
    main()

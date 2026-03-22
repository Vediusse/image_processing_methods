from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from image_lab2.io.config_loader import load_config
from image_lab2.report.exporters import export_json
from image_lab2.services.experiment_runner import ExperimentRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report"
GENERATED_DIR = REPORT_DIR / "generated"


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(PROJECT_ROOT / "examples" / "default_config.json")
    result = ExperimentRunner().run(config)
    export_json(result, GENERATED_DIR / "results.json")
    _write_values_typ(config, result, GENERATED_DIR / "values.typ")
    _plot_convergence(result, GENERATED_DIR / "convergence.png")
    _plot_error(result, GENERATED_DIR / "errors.png")


def _plot_convergence(result, destination: Path) -> None:
    figure = Figure(figsize=(10, 5), facecolor="white")
    axes = figure.add_subplot(111)
    for series in result.by_method.values():
        x_values = [estimate.sample_size for estimate in series.estimates]
        y_values = [estimate.estimate for estimate in series.estimates]
        axes.plot(x_values, y_values, marker="o", linewidth=1.4, label=series.display_name)
    axes.axhline(result.true_value, linestyle="--", color="black", linewidth=1.3, label="Истинный интеграл")
    axes.set_xscale("log")
    axes.set_xlabel("Размер выборки N")
    axes.set_ylabel("Оценка интеграла")
    axes.set_title("Сходимость методов Монте-Карло")
    axes.grid(alpha=0.25)
    axes.legend(fontsize=7, ncol=2)
    figure.savefig(destination, dpi=170, bbox_inches="tight")


def _plot_error(result, destination: Path) -> None:
    figure = Figure(figsize=(10, 5), facecolor="white")
    axes = figure.add_subplot(111)
    for series in result.by_method.values():
        x_values = [estimate.sample_size for estimate in series.estimates]
        y_values = [estimate.absolute_error for estimate in series.estimates]
        axes.plot(x_values, y_values, marker="o", linewidth=1.4, label=series.display_name)
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("Размер выборки N")
    axes.set_ylabel("Абсолютная ошибка")
    axes.set_title("Абсолютная ошибка методов")
    axes.grid(alpha=0.25)
    axes.legend(fontsize=7, ncol=2)
    figure.savefig(destination, dpi=170, bbox_inches="tight")


def _write_values_typ(config, result, destination: Path) -> None:
    rows = []
    for series in result.by_method.values():
        for estimate in series.estimates:
            rows.append(
                "  [{name}], [{n}], [{value:.8f}], [{abs_err:.8f}], [{rel_err:.8f}], [{stderr:.8f}],".format(
                    name=estimate.display_name,
                    n=estimate.sample_size,
                    value=estimate.estimate,
                    abs_err=estimate.absolute_error,
                    rel_err=estimate.relative_error,
                    stderr=estimate.estimated_error,
                )
            )
    best = min(
        [estimate for series in result.by_method.values() for estimate in series.estimates],
        key=lambda item: item.absolute_error,
    )
    content = """#let true_value = "{true_value:.8f}"
#let interval_label = "[{left:.1f}, {right:.1f}]"
#let sample_sizes_label = "{sample_sizes}"
#let strat_steps_label = "{strat_steps}"
#let is_powers_label = "{is_powers}"
#let mis_powers_label = "{mis_powers}"
#let rr_thresholds_label = "{rr_thresholds}"
#let error_formula = "{error_formula}"
#let best_method = "{best_method}"
#let best_n = "{best_n}"
#let best_estimate = "{best_estimate:.8f}"
#let best_abs_error = "{best_abs_error:.8f}"

#let results_table = table(
  columns: 6,
  inset: 6pt,
  stroke: (x, y) => if y == 0 {{ 0.9pt + rgb("#3f5974") }} else {{ 0.5pt + rgb("#8aa0b5") }},
  align: center,
  [Метод], [N], [Оценка], [Абс. ошибка], [Отн. ошибка], [Оценка погрешности],
{rows}
)
""".format(
        true_value=result.true_value,
        left=config.interval.left,
        right=config.interval.right,
        sample_sizes=", ".join(str(value) for value in config.sample_sizes),
        strat_steps=", ".join(str(value) for value in config.stratification_steps),
        is_powers=", ".join("x^{0}".format(value) for value in config.importance_sampling_powers),
        mis_powers=", ".join("x^{0}".format(value) for value in config.mis_powers),
        rr_thresholds=", ".join(str(value) for value in config.russian_roulette_thresholds),
        error_formula=result.estimated_error_formula,
        best_method=best.display_name,
        best_n=best.sample_size,
        best_estimate=best.estimate,
        best_abs_error=best.absolute_error,
        rows="\n".join(rows),
    )
    destination.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()

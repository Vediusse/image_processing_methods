from __future__ import annotations

from collections import OrderedDict

from image_lab2.math.functions import analytic_integral_x_squared
from image_lab2.models.config import ExperimentConfig
from image_lab2.models.results import ExperimentResult, MethodSeries
from image_lab2.services.estimators import MonteCarloEstimators


class ExperimentRunner:
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        if config.integrand != "x_squared":
            raise ValueError("Для ЛР2 реализована функция f(x)=x^2.")
        if config.interval.right <= config.interval.left:
            raise ValueError("Границы интервала заданы некорректно.")

        true_value = analytic_integral_x_squared(config.interval)
        estimators = MonteCarloEstimators(config)
        series = OrderedDict()

        for sample_size in config.sample_sizes:
            if sample_size <= 0:
                raise ValueError("Размер выборки должен быть положительным.")
            seed_base = config.seed + sample_size
            self._append(series, estimators.simple_mc(sample_size, seed_base + 1, true_value))
            for step_index, step in enumerate(config.stratification_steps, start=1):
                self._append(series, estimators.stratified(sample_size, step, seed_base + 10 + step_index, true_value))
            for power in config.importance_sampling_powers:
                self._append(series, estimators.importance_sampling(sample_size, power, seed_base + 100 + power, true_value))
            if len(config.mis_powers) != 2:
                raise ValueError("Для MIS ожидаются две плотности вероятности.")
            self._append(
                series,
                estimators.mis(sample_size, config.mis_powers[0], config.mis_powers[1], False, seed_base + 201, true_value),
            )
            self._append(
                series,
                estimators.mis(sample_size, config.mis_powers[0], config.mis_powers[1], True, seed_base + 202, true_value),
            )
            for index, threshold in enumerate(config.russian_roulette_thresholds, start=1):
                self._append(
                    series,
                    estimators.russian_roulette(sample_size, threshold, seed_base + 300 + index, true_value),
                )

        return ExperimentResult(
            true_value=true_value,
            estimated_error_formula="ΔI = I_true / sqrt(N)",
            by_method=series,
        )

    def _append(self, target, estimate) -> None:
        if estimate.method_key not in target:
            target[estimate.method_key] = MethodSeries(
                method_key=estimate.method_key,
                display_name=estimate.display_name,
                estimates=[],
            )
        target[estimate.method_key].estimates.append(estimate)

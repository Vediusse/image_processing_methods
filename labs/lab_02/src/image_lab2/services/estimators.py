from __future__ import annotations

import math
from typing import List

import numpy as np

from image_lab2.math.distributions import PowerPdf
from image_lab2.math.functions import analytic_integral_x_squared, evaluate_integrand
from image_lab2.models.config import ExperimentConfig, Interval
from image_lab2.models.results import MethodEstimate


def estimate_error(true_value: float, sample_size: int) -> float:
    return abs(true_value) / math.sqrt(sample_size)


def build_method_estimate(
    method_key: str,
    display_name: str,
    sample_size: int,
    true_value: float,
    estimate: float,
    estimated_error_value: float,
) -> MethodEstimate:
    absolute_error = abs(estimate - true_value)
    relative_error = absolute_error / abs(true_value) if true_value else 0.0
    return MethodEstimate(
        method_key=method_key,
        display_name=display_name,
        sample_size=sample_size,
        estimate=estimate,
        true_value=true_value,
        absolute_error=absolute_error,
        relative_error=relative_error,
        estimated_error=estimated_error_value,
    )


class MonteCarloEstimators:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.interval = config.interval

    def simple_mc(self, sample_size: int, seed: int, true_value: float) -> MethodEstimate:
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.interval.left, self.interval.right, sample_size)
        weighted = self.interval.width * evaluate_integrand(self.config.integrand, x)
        estimate = float(np.mean(weighted))
        stderr = self._standard_error(weighted)
        return build_method_estimate("simple_mc", "Простой MC", sample_size, true_value, estimate, stderr)

    def stratified(self, sample_size: int, step: float, seed: int, true_value: float) -> MethodEstimate:
        strata = self._build_strata(step)
        allocations = self._allocate_samples(sample_size, strata)
        rng = np.random.default_rng(seed)
        estimate = 0.0
        variance_sum = 0.0
        for allocation, interval in zip(allocations, strata):
            if allocation <= 0:
                continue
            x = rng.uniform(interval.left, interval.right, allocation)
            values = evaluate_integrand(self.config.integrand, x)
            weighted = interval.width * values
            estimate += float(np.mean(weighted))
            variance_sum += (interval.width**2) * self._integrand_variance(interval) / allocation
        stderr = math.sqrt(max(variance_sum, 0.0))
        key = "stratified_{0}".format(str(step).replace(".", "_"))
        display = "стратификация h={0}".format(step)
        return build_method_estimate(key, display, sample_size, true_value, estimate, stderr)

    def importance_sampling(self, sample_size: int, power: int, seed: int, true_value: float) -> MethodEstimate:
        rng = np.random.default_rng(seed)
        distribution = PowerPdf(self.interval, power)
        x = distribution.sample(rng, sample_size)
        weighted = evaluate_integrand(self.config.integrand, x) / distribution.pdf(x)
        estimate = float(np.mean(weighted))
        stderr = self._standard_error(weighted)
        key = "importance_x_pow_{0}".format(power)
        display = "Значимость p(x)~x^{0}".format(power)
        return build_method_estimate(key, display, sample_size, true_value, estimate, stderr)

    def mis(self, sample_size: int, power_a: int, power_b: int, use_power_heuristic: bool, seed: int, true_value: float):
        rng = np.random.default_rng(seed)
        count_a = sample_size // 2
        count_b = sample_size - count_a
        dist_a = PowerPdf(self.interval, power_a)
        dist_b = PowerPdf(self.interval, power_b)

        x_a = dist_a.sample(rng, count_a)
        x_b = dist_b.sample(rng, count_b)
        estimate = 0.0
        variance_terms = []
        if count_a:
            p_a = dist_a.pdf(x_a)
            p_b = dist_b.pdf(x_a)
            weights = self._mis_weights(p_a, p_b, first=True, squared=use_power_heuristic)
            contributions_a = weights * evaluate_integrand(self.config.integrand, x_a) / p_a
            estimate += float(np.mean(contributions_a))
            variance_terms.append(self._sample_variance(contributions_a) / count_a)
        if count_b:
            p_a = dist_a.pdf(x_b)
            p_b = dist_b.pdf(x_b)
            weights = self._mis_weights(p_a, p_b, first=False, squared=use_power_heuristic)
            contributions_b = weights * evaluate_integrand(self.config.integrand, x_b) / p_b
            estimate += float(np.mean(contributions_b))
            variance_terms.append(self._sample_variance(contributions_b) / count_b)
        stderr = math.sqrt(sum(variance_terms)) if variance_terms else 0.0
        key = "mis_power" if use_power_heuristic else "mis_balance"
        display = "MIS: средний квадрат" if use_power_heuristic else "MIS: средняя плотность"
        return build_method_estimate(key, display, sample_size, true_value, estimate, stderr)

    def russian_roulette(self, sample_size: int, threshold: float, seed: int, true_value: float) -> MethodEstimate:
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.interval.left, self.interval.right, sample_size)
        values = evaluate_integrand(self.config.integrand, x)
        max_value = self._max_integrand_on_interval()
        normalized = values / max_value
        survival = np.where(normalized >= threshold, 1.0, normalized / threshold)
        survival = np.clip(survival, 1e-12, 1.0)
        roulette = rng.random(sample_size)
        survived = roulette <= survival
        weighted = self.interval.width * np.where(survived, values / survival, 0.0)
        estimate = float(np.mean(weighted))
        stderr = self._standard_error(weighted)
        key = "russian_roulette_{0}".format(str(threshold).replace(".", "_"))
        display = "Русская рулетка R={0}".format(threshold)
        return build_method_estimate(key, display, sample_size, true_value, estimate, stderr)

    def _build_strata(self, step: float) -> List[Interval]:
        if step <= 0.0:
            raise ValueError("Шаг стратификации должен быть положительным.")
        left = self.interval.left
        right = self.interval.right
        edges = [left]
        while edges[-1] + step < right - 1e-12:
            edges.append(edges[-1] + step)
        edges.append(right)
        return [Interval(edges[index], edges[index + 1]) for index in range(len(edges) - 1)]

    def _allocate_samples(self, sample_size: int, strata: List[Interval]) -> List[int]:
        weights = np.asarray(
            [interval.width * max(math.sqrt(self._integrand_variance(interval)), 1e-12) for interval in strata],
            dtype=np.float64,
        )
        if sample_size <= 0:
            raise ValueError("Размер выборки должен быть положительным.")

        if sample_size >= len(strata):
            allocations = np.ones(len(strata), dtype=int)
            remaining = sample_size - len(strata)
        else:
            allocations = np.zeros(len(strata), dtype=int)
            remaining = sample_size

        if remaining > 0:
            scaled = weights / weights.sum() * remaining
            extra = np.floor(scaled).astype(int)
            allocations += extra
            leftover = remaining - int(extra.sum())
            if leftover > 0:
                order = np.argsort(-(scaled - extra))
                allocations[order[:leftover]] += 1
        return allocations.tolist()

    def _sample_variance(self, values) -> float:
        if len(values) <= 1:
            return 0.0
        return float(np.var(values, ddof=1))

    def _standard_error(self, weighted_values) -> float:
        if len(weighted_values) <= 1:
            return 0.0
        return float(np.std(weighted_values, ddof=1) / math.sqrt(len(weighted_values)))

    def _mis_weights(self, p_a, p_b, first: bool, squared: bool):
        if squared:
            a_term = p_a * p_a
            b_term = p_b * p_b
        else:
            a_term = p_a
            b_term = p_b
        denominator = a_term + b_term
        denominator = np.where(denominator <= 1e-16, 1e-16, denominator)
        if first:
            return a_term / denominator
        return b_term / denominator

    def _max_integrand_on_interval(self) -> float:
        candidates = [self.interval.left, self.interval.right]
        if self.interval.left <= 0.0 <= self.interval.right:
            candidates.append(0.0)
        values = evaluate_integrand(self.config.integrand, np.asarray(candidates, dtype=np.float64))
        return float(np.max(values))

    def _integrand_variance(self, interval: Interval) -> float:
        if self.config.integrand != "x_squared":
            raise ValueError("Для ЛР2 дисперсия стратификации реализована только для f(x)=x^2.")
        width = interval.width
        second_moment = analytic_integral_x_squared(interval) / width
        fourth_moment = ((interval.right**5) - (interval.left**5)) / (5.0 * width)
        variance = fourth_moment - second_moment * second_moment
        return max(float(variance), 0.0)

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import numpy as np


Rectangle = Tuple[str, float, float, float, float]


def triangle_rectangles(rectangle_count: int) -> List[Rectangle]:
    return _select_rectangles(_triangle_candidates, rectangle_count, radius=None, prefix="T")


def circle_rectangles(radius: float, rectangle_count: int) -> List[Rectangle]:
    return _select_rectangles(_circle_candidates, rectangle_count, radius=radius, prefix="C")


def rectangle_counts(
    values_x: np.ndarray,
    values_y: np.ndarray,
    rectangles: Iterable[Rectangle],
) -> Tuple[List[str], np.ndarray]:
    labels = []
    counts = []
    for name, x0, x1, y0, y1 in rectangles:
        labels.append(name)
        counts.append(int(np.sum((values_x >= x0) & (values_x <= x1) & (values_y >= y0) & (values_y <= y1))))
    return labels, np.asarray(counts, dtype=float)


def relative_spread(counts: np.ndarray) -> float:
    mean = float(np.mean(counts))
    if mean <= 1e-12:
        return 0.0
    return float(np.std(counts) / mean)


def _select_rectangles(candidate_builder, rectangle_count: int, radius: Optional[float], prefix: str) -> List[Rectangle]:
    if rectangle_count <= 0:
        raise ValueError("Количество прямоугольников должно быть положительным.")

    subdivisions = 1
    candidates: list[tuple[float, float, float, float]] = []
    while len(candidates) < rectangle_count:
        subdivisions += 1
        candidates = candidate_builder(subdivisions, radius)

    if len(candidates) == rectangle_count:
        selected = candidates
    else:
        selected = _spread_selection(candidates, rectangle_count)

    return [
        (f"{prefix}{index}", x0, x1, y0, y1)
        for index, (x0, x1, y0, y1) in enumerate(selected, start=1)
    ]


def _spread_selection(candidates: list[tuple[float, float, float, float]], rectangle_count: int):
    if rectangle_count >= len(candidates):
        return candidates
    indices = np.linspace(0, len(candidates) - 1, rectangle_count)
    selected_indices = []
    used = set()
    for index in np.round(indices).astype(int):
        candidate_index = int(index)
        while candidate_index in used and candidate_index + 1 < len(candidates):
            candidate_index += 1
        while candidate_index in used and candidate_index - 1 >= 0:
            candidate_index -= 1
        if candidate_index not in used:
            used.add(candidate_index)
            selected_indices.append(candidate_index)
    selected_indices.sort()
    return [candidates[index] for index in selected_indices[:rectangle_count]]


def _triangle_candidates(subdivisions: int, _radius: Optional[float]):
    step = 1.0 / subdivisions
    rectangles = []
    for row in range(subdivisions):
        for col in range(subdivisions):
            u0 = col * step
            u1 = (col + 1) * step
            v0 = row * step
            v1 = (row + 1) * step
            if u1 + v1 <= 1.0 + 1e-12:
                rectangles.append((u0, u1, v0, v1))
    rectangles.sort(key=lambda item: ((item[2] + item[3]) * 0.5, (item[0] + item[1]) * 0.5))
    return rectangles


def _circle_candidates(subdivisions: int, radius: Optional[float]):
    if radius is None:
        raise ValueError("Для круга нужен радиус.")
    step = (2.0 * radius) / subdivisions
    rectangles = []
    for row in range(subdivisions):
        for col in range(subdivisions):
            x0 = -radius + col * step
            x1 = x0 + step
            y0 = -radius + row * step
            y1 = y0 + step
            if _rectangle_inside_circle(x0, x1, y0, y1, radius):
                rectangles.append((x0, x1, y0, y1))
    rectangles.sort(key=lambda item: ((item[2] + item[3]) * 0.5, (item[0] + item[1]) * 0.5))
    return rectangles


def _rectangle_inside_circle(x0: float, x1: float, y0: float, y1: float, radius: float) -> bool:
    corners = ((x0, y0), (x0, y1), (x1, y0), (x1, y1))
    return all(math.hypot(x, y) <= radius + 1e-12 for x, y in corners)

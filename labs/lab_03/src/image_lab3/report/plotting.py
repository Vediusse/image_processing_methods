from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure


def create_distribution_figure(distribution, facecolor: str = "#10141c") -> Figure:
    figure = Figure(figsize=(9, 5), facecolor=facecolor)
    scatter_axes = figure.add_subplot(121, projection="3d")
    hist_axes = figure.add_subplot(122)
    figure.subplots_adjust(left=0.05, right=0.97, bottom=0.10, top=0.92, wspace=0.20)
    plot_distribution_on_axes(scatter_axes, hist_axes, distribution)
    return figure


def plot_distribution_on_axes(scatter_axes, hist_axes, distribution) -> None:
    scatter_axes.clear()
    hist_axes.clear()
    _style_axes(scatter_axes, hist_axes)

    points = distribution.points
    sample = points[: min(7000, len(points))]
    scatter_axes.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=2, alpha=0.45, color="#41c7ff")
    scatter_axes.set_title(distribution.title, color="#e9f1fb")
    scatter_axes.set_xlabel("X")
    scatter_axes.set_ylabel("Y")
    scatter_axes.set_zlabel("Z")
    _set_equal_limits(scatter_axes, sample)

    if distribution.key == "triangle":
        alpha = distribution.auxiliary["alpha"]
        hist_axes.hist(alpha, bins=20, range=(0.0, 1.0), density=True, color="#71d8ff", alpha=0.78, label="эмпирическое")
        x = np.linspace(0.0, 1.0, 200)
        hist_axes.plot(x, 2.0 * (1.0 - x), color="#f1fa8c", linewidth=2.0, label="теория 2(1-α)")
        hist_axes.set_title("Распределение барицентрической координаты α", color="#e9f1fb")
        hist_axes.set_xlabel("α")
        hist_axes.set_ylabel("Плотность")
        hist_axes.legend(fontsize=8, loc="upper right")
    elif distribution.key == "circle":
        values = distribution.auxiliary["radius_squared_normalized"]
        hist_axes.hist(values, bins=20, range=(0.0, 1.0), density=True, color="#50fa7b", alpha=0.85, label="эмпирическое")
        hist_axes.plot([0.0, 1.0], [1.0, 1.0], color="#f1fa8c", linewidth=2.0, label="теория 1")
        hist_axes.set_title("Распределение r²/R²", color="#e9f1fb")
        hist_axes.set_xlabel("r²/R²")
        hist_axes.set_ylabel("Плотность")
        hist_axes.legend(fontsize=8, loc="upper right")
    elif distribution.key == "sphere":
        values = distribution.auxiliary["z"]
        hist_axes.hist(values, bins=20, range=(-1.0, 1.0), density=True, color="#ffb86c", alpha=0.85, label="эмпирическое")
        hist_axes.plot([-1.0, 1.0], [0.5, 0.5], color="#f1fa8c", linewidth=2.0, label="теория 1/2")
        hist_axes.set_title("Распределение z", color="#e9f1fb")
        hist_axes.set_xlabel("z")
        hist_axes.set_ylabel("Плотность")
        hist_axes.legend(fontsize=8, loc="upper right")
    else:
        cosine = distribution.auxiliary["cosine"]
        hist_axes.hist(cosine, bins=20, density=True, color="#ff79c6", alpha=0.75, label="эмпирическое")
        x = np.linspace(0.0, 1.0, 200)
        hist_axes.plot(x, 2.0 * x, color="#f1fa8c", linewidth=2.0, label="теория 2μ")
        hist_axes.set_title("Распределение cos(theta)", color="#e9f1fb")
        hist_axes.set_xlabel("μ = cos(theta)")
        hist_axes.set_ylabel("Плотность")
        hist_axes.legend(fontsize=8, loc="upper left")

    hist_axes.tick_params(colors="#dde7f2")
    hist_axes.xaxis.label.set_color("#dde7f2")
    hist_axes.yaxis.label.set_color("#dde7f2")
    hist_axes.title.set_color("#dde7f2")
    hist_axes.grid(alpha=0.2)


def _style_axes(scatter_axes, hist_axes) -> None:
    for axes in (scatter_axes, hist_axes):
        axes.set_facecolor("#10141c")
        axes.tick_params(colors="#dde7f2")
        axes.xaxis.label.set_color("#dde7f2")
        axes.yaxis.label.set_color("#dde7f2")
        axes.title.set_color("#dde7f2")
        for spine in getattr(axes, "spines", {}).values():
            spine.set_color("#36526f")


def _set_equal_limits(axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(np.max(maxs - mins) / 2.0, 1e-6)
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)

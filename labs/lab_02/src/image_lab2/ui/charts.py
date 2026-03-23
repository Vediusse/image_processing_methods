from __future__ import annotations

from matplotlib.lines import Line2D
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from image_lab2.models.results import ExperimentResult


class ChartsCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(10, 7), facecolor="#f6f1e8")
        super().__init__(self.figure)
        grid = self.figure.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.38, wspace=0.22)
        self.axes_estimate = self.figure.add_subplot(grid[0, :])
        self.axes_error = self.figure.add_subplot(grid[1, 0])
        self.axes_stderr = self.figure.add_subplot(grid[1, 1])
        self.figure.subplots_adjust(left=0.07, right=0.82, top=0.94, bottom=0.09)
        self._legend = None
        self._style_axes()

    def _style_axes(self) -> None:
        for axes in (self.axes_estimate, self.axes_error, self.axes_stderr):
            axes.set_facecolor("#fffdf8")
            axes.tick_params(colors="#17323a")
            axes.xaxis.label.set_color("#17323a")
            axes.yaxis.label.set_color("#17323a")
            axes.title.set_color("#17323a")
            for spine in axes.spines.values():
                spine.set_color("#d7c9b2")
            axes.grid(color="#d8ccb6", alpha=0.45, linewidth=0.8)

    def plot_result(self, result: ExperimentResult) -> None:
        self.axes_estimate.clear()
        self.axes_error.clear()
        self.axes_stderr.clear()
        self._style_axes()
        if self._legend is not None:
            self._legend.remove()
            self._legend = None

        legend_handles = []
        for series in result.by_method.values():
            x_values = [estimate.sample_size for estimate in series.estimates]
            y_values = [estimate.estimate for estimate in series.estimates]
            error_values = [estimate.absolute_error for estimate in series.estimates]
            stderr_values = [estimate.estimated_error for estimate in series.estimates]
            estimate_line, = self.axes_estimate.plot(x_values, y_values, marker="o", linewidth=2.0, markersize=5)
            self.axes_error.plot(x_values, error_values, marker="o", linewidth=1.8, markersize=5, color=estimate_line.get_color())
            self.axes_stderr.plot(
                x_values,
                stderr_values,
                marker="o",
                linewidth=1.8,
                markersize=5,
                color=estimate_line.get_color(),
            )
            legend_handles.append(
                Line2D([0], [0], color=estimate_line.get_color(), marker="o", linewidth=1.8, label=series.display_name)
            )

        self.axes_estimate.axhline(result.true_value, color="#d66b4d", linestyle="--", linewidth=1.5)
        legend_handles.append(Line2D([0], [0], color="#d66b4d", linestyle="--", linewidth=1.5, label="Истинный интеграл"))
        self.axes_estimate.set_xscale("log")
        self.axes_error.set_xscale("log")
        self.axes_error.set_yscale("log")
        self.axes_stderr.set_xscale("log")
        self.axes_stderr.set_yscale("log")
        self.axes_estimate.set_title("Карта сходимости оценок")
        self.axes_error.set_title("Абсолютная ошибка")
        self.axes_stderr.set_title("Оценка стандартной ошибки")
        self.axes_estimate.set_xlabel("Размер выборки N")
        self.axes_estimate.set_ylabel("Оценка интеграла")
        self.axes_error.set_xlabel("Размер выборки N")
        self.axes_error.set_ylabel("|I - I_true|")
        self.axes_stderr.set_xlabel("Размер выборки N")
        self.axes_stderr.set_ylabel("stderr")
        self._legend = self.figure.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.835, 0.5),
            fontsize=8,
            frameon=True,
        )
        self._legend.get_frame().set_facecolor("#fffaf2")
        self._legend.get_frame().set_edgecolor("#d7c9b2")
        self.draw()

from __future__ import annotations

from matplotlib.lines import Line2D
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from image_lab2.models.results import ExperimentResult


class ChartsCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(8, 6), facecolor="#10141c")
        super().__init__(self.figure)
        self.axes_estimate = self.figure.add_subplot(211)
        self.axes_error = self.figure.add_subplot(212)
        self.figure.subplots_adjust(hspace=0.42, left=0.08, right=0.78, top=0.95, bottom=0.08)
        self._legend = None
        self._style_axes()

    def _style_axes(self) -> None:
        for axes in (self.axes_estimate, self.axes_error):
            axes.set_facecolor("#10141c")
            axes.tick_params(colors="#dde7f2")
            axes.xaxis.label.set_color("#dde7f2")
            axes.yaxis.label.set_color("#dde7f2")
            axes.title.set_color("#dde7f2")
            for spine in axes.spines.values():
                spine.set_color("#36526f")

    def plot_result(self, result: ExperimentResult) -> None:
        self.axes_estimate.clear()
        self.axes_error.clear()
        self._style_axes()
        if self._legend is not None:
            self._legend.remove()
            self._legend = None

        legend_handles = []
        for series in result.by_method.values():
            x_values = [estimate.sample_size for estimate in series.estimates]
            y_values = [estimate.estimate for estimate in series.estimates]
            error_values = [estimate.absolute_error for estimate in series.estimates]
            estimate_line, = self.axes_estimate.plot(x_values, y_values, marker="o", linewidth=1.5)
            self.axes_error.plot(x_values, error_values, marker="o", linewidth=1.5, color=estimate_line.get_color())
            legend_handles.append(
                Line2D([0], [0], color=estimate_line.get_color(), marker="o", linewidth=1.5, label=series.display_name)
            )

        self.axes_estimate.axhline(result.true_value, color="#ffd166", linestyle="--", linewidth=1.3)
        legend_handles.append(Line2D([0], [0], color="#ffd166", linestyle="--", linewidth=1.3, label="Истинный интеграл"))
        self.axes_estimate.set_xscale("log")
        self.axes_error.set_xscale("log")
        self.axes_error.set_yscale("log")
        self.axes_estimate.set_title("Сходимость оценки интеграла")
        self.axes_error.set_title("Абсолютная ошибка")
        self.axes_estimate.set_xlabel("Размер выборки N")
        self.axes_estimate.set_ylabel("Оценка интеграла")
        self.axes_error.set_xlabel("Размер выборки N")
        self.axes_error.set_ylabel("|I - I_true|")
        self.axes_estimate.grid(alpha=0.2)
        self.axes_error.grid(alpha=0.2)
        self._legend = self.figure.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.80, 0.5),
            fontsize=7,
            frameon=True,
        )
        self.draw()

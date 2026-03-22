from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from image_lab3.report.plotting import plot_distribution_on_axes


class DistributionCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(9, 5), facecolor="#10141c")
        super().__init__(self.figure)
        self.axes_scatter = self.figure.add_subplot(121, projection="3d")
        self.axes_hist = self.figure.add_subplot(122)
        self.figure.subplots_adjust(left=0.05, right=0.97, bottom=0.10, top=0.92, wspace=0.20)

    def plot_distribution(self, distribution) -> None:
        plot_distribution_on_axes(self.axes_scatter, self.axes_hist, distribution)
        self.draw()

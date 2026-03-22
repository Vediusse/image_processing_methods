from __future__ import annotations

from typing import Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from image_lab1.models.results import BrightnessResult
from image_lab1.models.scene import Scenario


class VisualizationCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(7, 5), facecolor="#10141c")
        super().__init__(self.figure)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self._apply_style()

    def _apply_style(self) -> None:
        self.axes.set_facecolor("#10141c")
        self.figure.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.96)
        self.axes.xaxis.label.set_color("#dde7f2")
        self.axes.yaxis.label.set_color("#dde7f2")
        self.axes.zaxis.label.set_color("#dde7f2")
        self.axes.tick_params(colors="#dde7f2")
        self.axes.set_title("Визуализация сцены", color="#dde7f2")

    def plot_scene(self, scenario: Scenario, result: Optional[BrightnessResult] = None) -> None:
        self.axes.clear()
        self._apply_style()

        triangle = scenario.triangle
        face_color = (0.12, 0.48, 0.72, 0.35)
        if result and result.local_evaluations:
            average = self._average_brightness(result)
            face_color = (average[0], average[1], average[2], 0.45)

        triangle_surface = Poly3DCollection(
            [[triangle.a.to_tuple(), triangle.b.to_tuple(), triangle.c.to_tuple()]],
            facecolors=[face_color],
            edgecolors="#67d4ff",
            linewidths=2.2,
        )
        self.axes.add_collection3d(triangle_surface)

        triangle_points = [
            triangle.a.to_tuple(),
            triangle.b.to_tuple(),
            triangle.c.to_tuple(),
            triangle.a.to_tuple(),
        ]
        xs, ys, zs = zip(*triangle_points)
        self.axes.plot(xs, ys, zs, color="#67d4ff", linewidth=2.5)
        self.axes.scatter(xs[:-1], ys[:-1], zs[:-1], color="#67d4ff", s=35, label="Треугольник")
        for name, point in (("A", triangle.a), ("B", triangle.b), ("C", triangle.c)):
            self.axes.text(point.x, point.y, point.z, " " + name, color="#d7f3ff")

        for index, light in enumerate(scenario.lights, start=1):
            self.axes.scatter(
                [light.position.x],
                [light.position.y],
                [light.position.z],
                color=(light.intensity.r, light.intensity.g, light.intensity.b),
                s=80,
                marker="*",
            )
            axis = light.axis_direction.normalized() * 0.7
            self.axes.quiver(
                light.position.x,
                light.position.y,
                light.position.z,
                axis.x,
                axis.y,
                axis.z,
                color="#ffe082",
            )
            self.axes.text(
                light.position.x,
                light.position.y,
                light.position.z,
                " Источник {idx}".format(idx=index),
                color="#ffe8a3",
            )

        observer = scenario.observer.position
        self.axes.scatter([observer.x], [observer.y], [observer.z], color="#f06292", s=45, marker="s")
        self.axes.text(observer.x, observer.y, observer.z, " Камера", color="#ff9bc2")

        points = result.local_evaluations if result else []
        for item in points:
            brightness = item.total_brightness.clamp()
            self.axes.scatter(
                [item.global_point.x],
                [item.global_point.y],
                [item.global_point.z],
                color=(brightness.r, brightness.g, brightness.b),
                s=60,
            )
            normal = item.normal * 0.5
            self.axes.quiver(
                item.global_point.x,
                item.global_point.y,
                item.global_point.z,
                normal.x,
                normal.y,
                normal.z,
                color="#8bc34a",
            )
            self.axes.text(
                item.global_point.x,
                item.global_point.y,
                item.global_point.z,
                " {name}".format(name=item.point_name),
                color="#ffffff",
            )
            self.axes.text(
                item.global_point.x + normal.x,
                item.global_point.y + normal.y,
                item.global_point.z + normal.z,
                " n",
                color="#a8e06d",
            )
            for contribution in item.contributions:
                self.axes.plot(
                    [item.global_point.x, contribution.direction_to_light.x * contribution.distance + item.global_point.x],
                    [item.global_point.y, contribution.direction_to_light.y * contribution.distance + item.global_point.y],
                    [item.global_point.z, contribution.direction_to_light.z * contribution.distance + item.global_point.z],
                    color="#ffd166",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.35,
                )

        self.axes.set_xlabel("X")
        self.axes.set_ylabel("Y")
        self.axes.set_zlabel("Z")
        self._set_equal_limits(scenario, result)
        self.axes.view_init(elev=22, azim=-72)
        self.draw()

    def _average_brightness(self, result: BrightnessResult) -> tuple[float, float, float]:
        count = max(len(result.local_evaluations), 1)
        r = sum(item.total_brightness.r for item in result.local_evaluations) / count
        g = sum(item.total_brightness.g for item in result.local_evaluations) / count
        b = sum(item.total_brightness.b for item in result.local_evaluations) / count
        max_channel = max(r, g, b, 1e-6)
        scale = min(1.0, 0.85 / max_channel)
        return (r * scale, g * scale, b * scale)

    def _set_equal_limits(self, scenario: Scenario, result: Optional[BrightnessResult]) -> None:
        points = [
            scenario.triangle.a,
            scenario.triangle.b,
            scenario.triangle.c,
            scenario.observer.position,
            *[light.position for light in scenario.lights],
        ]
        if result:
            points.extend(item.global_point for item in result.local_evaluations)

        xs = [point.x for point in points]
        ys = [point.y for point in points]
        zs = [point.z for point in points]
        center_x = (min(xs) + max(xs)) / 2.0
        center_y = (min(ys) + max(ys)) / 2.0
        center_z = (min(zs) + max(zs)) / 2.0
        radius = max(
            max(xs) - min(xs),
            max(ys) - min(ys),
            max(zs) - min(zs),
            1.0,
        ) / 2.0 + 0.5
        self.axes.set_xlim(center_x - radius, center_x + radius)
        self.axes.set_ylim(center_y - radius, center_y + radius)
        self.axes.set_zlim(center_z - radius, center_z + radius)

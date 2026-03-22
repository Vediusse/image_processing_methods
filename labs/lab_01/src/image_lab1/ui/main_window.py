from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from image_lab1.io.scenario_loader import load_scenario, save_scenario
from image_lab1.math.geometry import local_to_global
from image_lab1.models.results import BrightnessResult, PointEvaluation
from image_lab1.models.scene import (
    Camera,
    ColorRGB,
    Light,
    LocalPoint,
    Material,
    RadiationProfile,
    Scenario,
    Triangle,
)
from image_lab1.models.vector import Point3, Vector3
from image_lab1.report.exporters import export_csv, export_json, export_markdown
from image_lab1.services.observer_renderer import ObserverRenderer
from image_lab1.services.scenario_runner import ScenarioRunner
from image_lab1.ui.render_view import RenderView
from image_lab1.ui.visualization import VisualizationCanvas


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.runner = ScenarioRunner()
        self.renderer = ObserverRenderer(width=720, height=720, fov_degrees=34.0)
        self._updating_form = False
        self.current_result: Optional[BrightnessResult] = None
        self.current_scenario: Scenario = self._demo_scenario()
        self.setWindowTitle("ЛР 1 - Методы обработки изображений")
        self.resize(1660, 980)
        self._build_ui()
        self._load_scenario_into_form(self.current_scenario)
        self.visualization.plot_scene(self.current_scenario, None)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)

        button_row = QHBoxLayout()
        self.demo_button = QPushButton("Демо")
        self.calculate_button = QPushButton("Рассчитать")
        self.save_button = QPushButton("Сохранить сценарий")
        self.load_button = QPushButton("Загрузить сценарий")
        self.export_button = QPushButton("Экспорт результатов")
        for button in (
            self.demo_button,
            self.calculate_button,
            self.save_button,
            self.load_button,
            self.export_button,
        ):
            button.setMinimumHeight(36)
            button_row.addWidget(button)
        left_layout.addLayout(button_row)

        self.points_hint = QLabel(
            "Редактируйте только локальные координаты (u, v). "
            "Глобальные координаты ниже пересчитываются автоматически по треугольнику."
        )
        self.points_hint.setWordWrap(True)
        self.points_hint.setStyleSheet(
            "QLabel { background: #101925; border: 1px solid #1f3447; border-radius: 12px; padding: 10px; color: #b8d8f1; }"
        )

        left_layout.addWidget(self._build_triangle_box())
        left_layout.addWidget(self._build_material_box())
        left_layout.addWidget(self._build_observer_box())
        left_layout.addWidget(self._build_lights_box())
        left_layout.addWidget(self._build_points_box())
        left_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)

        self.visualization = VisualizationCanvas()
        self.results_tabs = QTabWidget()
        self.local_results_table = self._build_results_table()
        self.global_results_table = self._build_results_table()
        self.details_output = QTextEdit()
        self.details_output.setReadOnly(True)
        self.details_output.setMinimumHeight(180)
        self.render_view = RenderView()
        self.help_output = QTextEdit()
        self.help_output.setReadOnly(True)
        self.help_output.setMinimumHeight(180)

        self.results_tabs.addTab(self.render_view, "Рендер наблюдателя")
        self.results_tabs.addTab(self.local_results_table, "Точки: локальные")
        self.results_tabs.addTab(self.global_results_table, "Точки: глобальные")
        self.results_tabs.addTab(self.details_output, "Подробный отчет")
        self.results_tabs.addTab(self.help_output, "Что считается")

        self.scene_summary = QLabel()
        self.scene_summary.setWordWrap(True)
        self.scene_summary.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.scene_summary.setStyleSheet(
            "QLabel { background: #101925; border: 1px solid #1f3447; border-radius: 12px; padding: 10px; }"
        )

        right_layout.addWidget(self.visualization, stretch=3)
        right_layout.addWidget(self.scene_summary, stretch=0)
        right_layout.addWidget(self.results_tabs, stretch=2)

        left_scroll = self._wrap_scroll_area(left_panel)
        right_scroll = self._wrap_scroll_area(right_panel)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_scroll)
        splitter.setSizes([760, 900])

        self.demo_button.clicked.connect(self._on_load_demo)
        self.calculate_button.clicked.connect(self._on_calculate)
        self.save_button.clicked.connect(self._on_save_scenario)
        self.load_button.clicked.connect(self._on_load_scenario)
        self.export_button.clicked.connect(self._on_export_results)
        self._connect_form_events()

        self._apply_styles()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b1118; color: #e9f1fb; font-size: 13px; }
            QGroupBox {
                border: 1px solid #1f3447;
                border-radius: 14px;
                margin-top: 12px;
                padding-top: 12px;
                background: #101925;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #9ed8ff;
                font-weight: 600;
            }
            QLineEdit, QTextEdit, QTableWidget, QTabWidget::pane {
                background: #132131;
                border: 1px solid #22425e;
                border-radius: 10px;
                padding: 6px;
                selection-background-color: #2a5677;
            }
            QScrollArea { border: none; }
            QPushButton {
                background: #17324a;
                border: 1px solid #285a84;
                border-radius: 10px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background: #1d4564; }
            QTableWidget {
                gridline-color: #22425e;
            }
            QHeaderView::section {
                background: #16324b;
                color: #f3f8ff;
                border: none;
                padding: 6px;
            }
            QTabBar::tab {
                background: #122030;
                padding: 8px 14px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #1d4564;
            }
            """
        )

    def _build_triangle_box(self) -> QGroupBox:
        box = QGroupBox("Вершины треугольника")
        grid = QGridLayout(box)
        self.triangle_inputs: dict[str, list[QLineEdit]] = {}
        for row, vertex_name in enumerate(("a", "b", "c")):
            grid.addWidget(QLabel(vertex_name.upper()), row, 0)
            fields: list[QLineEdit] = []
            for col, axis in enumerate(("x", "y", "z"), start=1):
                field = QLineEdit()
                field.setPlaceholderText(axis)
                fields.append(field)
                grid.addWidget(field, row, col)
            self.triangle_inputs[vertex_name] = fields
        return box

    def _build_material_box(self) -> QGroupBox:
        box = QGroupBox("Материал поверхности")
        form = QFormLayout(box)
        self.material_color = [QLineEdit() for _ in range(3)]
        color_row = self._vector_row(self.material_color, ("R", "G", "B"))
        self.kd_input = QLineEdit()
        self.ks_input = QLineEdit()
        self.shininess_input = QLineEdit()
        form.addRow("Цвет поверхности", color_row)
        form.addRow("Коэффициент диффузии", self.kd_input)
        form.addRow("Коэффициент зеркальности", self.ks_input)
        form.addRow("Ширина блика (shininess)", self.shininess_input)
        return box

    def _build_observer_box(self) -> QGroupBox:
        box = QGroupBox("Наблюдатель / камера")
        form = QFormLayout(box)
        self.observer_inputs = [QLineEdit() for _ in range(3)]
        form.addRow("Положение", self._vector_row(self.observer_inputs, ("X", "Y", "Z")))
        return box

    def _build_lights_box(self) -> QGroupBox:
        box = QGroupBox("Источники света")
        layout = QVBoxLayout(box)
        self.lights_table = QTableWidget(0, 12)
        self.lights_table.setHorizontalHeaderLabels(
            [
                "Имя",
                "X",
                "Y",
                "Z",
                "Ось X",
                "Ось Y",
                "Ось Z",
                "R",
                "G",
                "B",
                "Степень",
                "Срез, °",
            ]
        )
        self._configure_input_table(self.lights_table, min_height=210)
        self.lights_table.setColumnWidth(0, 140)
        for column in range(1, 12):
            self.lights_table.setColumnWidth(column, 92)
        layout.addWidget(self.lights_table)
        return box

    def _build_points_box(self) -> QGroupBox:
        box = QGroupBox("Точки расчета")
        layout = QVBoxLayout(box)
        layout.addWidget(self.points_hint)
        self.local_points_table = QTableWidget(0, 3)
        self.local_points_table.setHorizontalHeaderLabels(["Имя", "u", "v"])
        self.global_points_table = QTableWidget(0, 4)
        self.global_points_table.setHorizontalHeaderLabels(["Имя", "x", "y", "z"])
        self._configure_input_table(self.local_points_table, min_height=150)
        self._configure_readonly_table(self.global_points_table, min_height=160)
        layout.addWidget(QLabel("Локальные координаты"))
        layout.addWidget(self.local_points_table)
        layout.addWidget(QLabel("Глобальные координаты"))
        layout.addWidget(self.global_points_table)
        return box

    def _build_results_table(self) -> QTableWidget:
        table = QTableWidget(0, 7)
        table.setHorizontalHeaderLabels(
            ["Точка", "Лок. (u, v)", "Глоб. (x, y, z)", "Нормаль", "Освещенность RGB", "Яркость RGB", "Вклады"]
        )
        self._configure_result_table(table)
        return table

    def _configure_input_table(self, table: QTableWidget, min_height: int) -> None:
        table.setMinimumHeight(min_height)
        table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.AnyKeyPressed
            | QAbstractItemView.SelectedClicked
        )
        table.setSelectionBehavior(QAbstractItemView.SelectItems)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setWordWrap(False)
        table.verticalHeader().setDefaultSectionSize(32)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.horizontalHeader().setMinimumSectionSize(80)

    def _configure_result_table(self, table: QTableWidget) -> None:
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setWordWrap(False)
        table.setMinimumHeight(260)
        table.verticalHeader().setDefaultSectionSize(34)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)

    def _configure_readonly_table(self, table: QTableWidget, min_height: int) -> None:
        table.setMinimumHeight(min_height)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setWordWrap(False)
        table.verticalHeader().setDefaultSectionSize(32)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.horizontalHeader().setMinimumSectionSize(90)

    def _connect_form_events(self) -> None:
        for edits in self.triangle_inputs.values():
            for edit in edits:
                edit.editingFinished.connect(self._refresh_global_points_preview)
        self.local_points_table.itemChanged.connect(self._handle_local_points_changed)

    def _handle_local_points_changed(self, _) -> None:
        if self._updating_form:
            return
        self._refresh_global_points_preview()

    def _wrap_scroll_area(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setWidget(widget)
        return area

    def _vector_row(self, edits: list[QLineEdit], placeholders: tuple[str, str, str]) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        for edit, placeholder in zip(edits, placeholders):
            edit.setPlaceholderText(placeholder)
            layout.addWidget(edit)
        return widget

    def _on_load_demo(self) -> None:
        self.current_scenario = self._demo_scenario()
        self._load_scenario_into_form(self.current_scenario)
        self.current_result = None
        self.visualization.plot_scene(self.current_scenario, None)
        self._clear_results()

    def _on_calculate(self) -> None:
        try:
            self.current_scenario = self._scenario_from_form()
            self.current_result = self.runner.run(self.current_scenario)
            self._show_result(self.current_result)
            self.visualization.plot_scene(self.current_scenario, self.current_result)
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка расчета", str(error))

    def _on_save_scenario(self) -> None:
        try:
            scenario = self._scenario_from_form()
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить сценарий", "", "JSON (*.json)")
            if path:
                save_scenario(path, scenario)
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка сохранения", str(error))

    def _on_load_scenario(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить сценарий", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.current_scenario = load_scenario(path)
            self._load_scenario_into_form(self.current_scenario)
            self.current_result = None
            self.visualization.plot_scene(self.current_scenario, None)
            self._clear_results()
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка загрузки", str(error))

    def _on_export_results(self) -> None:
        if self.current_result is None:
            QMessageBox.information(self, "Экспорт", "Сначала выполните расчет.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Экспорт результатов")
        if not directory:
            return
        export_dir = Path(directory)
        export_json(self.current_result, export_dir / "lab1_results.json")
        export_csv(self.current_result, export_dir / "lab1_results.csv")
        export_markdown(self.current_result, export_dir / "lab1_results.md")
        QMessageBox.information(self, "Экспорт", f"Результаты сохранены в {export_dir}")

    def _scenario_from_form(self) -> Scenario:
        triangle = Triangle(
            a=self._read_point(self.triangle_inputs["a"]),
            b=self._read_point(self.triangle_inputs["b"]),
            c=self._read_point(self.triangle_inputs["c"]),
        )
        material = Material(
            color=ColorRGB(*(self._read_float(edit) for edit in self.material_color)),
            diffuse_coefficient=self._read_float(self.kd_input),
            specular_coefficient=self._read_float(self.ks_input),
            shininess=self._read_float(self.shininess_input),
        )
        observer = Camera(position=self._read_point(self.observer_inputs))
        lights: list[Light] = []
        for row in range(self.lights_table.rowCount()):
            lights.append(
                Light(
                    name=self._table_text(self.lights_table, row, 0),
                    position=Point3(
                        self._table_float(self.lights_table, row, 1),
                        self._table_float(self.lights_table, row, 2),
                        self._table_float(self.lights_table, row, 3),
                    ),
                    axis_direction=Vector3(
                        self._table_float(self.lights_table, row, 4),
                        self._table_float(self.lights_table, row, 5),
                        self._table_float(self.lights_table, row, 6),
                    ),
                    intensity=ColorRGB(
                        self._table_float(self.lights_table, row, 7),
                        self._table_float(self.lights_table, row, 8),
                        self._table_float(self.lights_table, row, 9),
                    ),
                    profile=RadiationProfile(
                        exponent=self._table_float(self.lights_table, row, 10),
                        cutoff_degrees=self._table_float(self.lights_table, row, 11),
                    ),
                )
            )
        local_points = [
            LocalPoint(
                name=self._table_text(self.local_points_table, row, 0),
                u=self._table_float(self.local_points_table, row, 1),
                v=self._table_float(self.local_points_table, row, 2),
            )
            for row in range(self.local_points_table.rowCount())
        ]
        global_points = {
            point.name: local_to_global(triangle, point.u, point.v)
            for point in local_points
        }
        return Scenario(
            triangle=triangle,
            lights=lights,
            material=material,
            observer=observer,
            local_points=local_points,
            global_points=global_points,
        )

    def _load_scenario_into_form(self, scenario: Scenario) -> None:
        self._updating_form = True
        self._set_point(self.triangle_inputs["a"], scenario.triangle.a)
        self._set_point(self.triangle_inputs["b"], scenario.triangle.b)
        self._set_point(self.triangle_inputs["c"], scenario.triangle.c)
        self._set_color(self.material_color, scenario.material.color)
        self.kd_input.setText(str(scenario.material.diffuse_coefficient))
        self.ks_input.setText(str(scenario.material.specular_coefficient))
        self.shininess_input.setText(str(scenario.material.shininess))
        self._set_point(self.observer_inputs, scenario.observer.position)

        self.lights_table.setRowCount(len(scenario.lights))
        for row, light in enumerate(scenario.lights):
            values = [
                light.name,
                light.position.x,
                light.position.y,
                light.position.z,
                light.axis_direction.x,
                light.axis_direction.y,
                light.axis_direction.z,
                light.intensity.r,
                light.intensity.g,
                light.intensity.b,
                light.profile.exponent,
                light.profile.cutoff_degrees,
            ]
            for col, value in enumerate(values):
                self.lights_table.setItem(row, col, QTableWidgetItem(str(value)))

        self.local_points_table.setRowCount(len(scenario.local_points))
        for row, point in enumerate(scenario.local_points):
            for col, value in enumerate((point.name, point.u, point.v)):
                self.local_points_table.setItem(row, col, QTableWidgetItem(str(value)))

        self._updating_form = False
        self._refresh_global_points_preview()

    def _show_result(self, result: BrightnessResult) -> None:
        render_result = self.renderer.render(self.current_scenario)
        self._fill_results_table(self.local_results_table, result.local_evaluations)
        self._fill_results_table(self.global_results_table, result.global_evaluations)
        self.details_output.setPlainText(self._build_details_text(result))
        self.render_view.set_render_result(render_result)
        self.help_output.setPlainText(self._build_help_text())
        self.scene_summary.setText(self._build_scene_summary(result))

    def _fill_results_table(self, table: QTableWidget, evaluations: list[PointEvaluation]) -> None:
        table.setRowCount(len(evaluations))
        for row, item in enumerate(evaluations):
            lights_text = "; ".join(
                f"{contribution.light_name}: E={contribution.colored_irradiance.r:.4f}/{contribution.colored_irradiance.g:.4f}/{contribution.colored_irradiance.b:.4f}, "
                f"L={contribution.brightness.r:.4f}/{contribution.brightness.g:.4f}/{contribution.brightness.b:.4f}"
                for contribution in item.contributions
            )
            values = [
                item.point_name,
                f"({item.local_coordinates[0]:.3f}, {item.local_coordinates[1]:.3f})",
                f"({item.global_point.x:.3f}, {item.global_point.y:.3f}, {item.global_point.z:.3f})",
                f"({item.normal.x:.3f}, {item.normal.y:.3f}, {item.normal.z:.3f})",
                f"({item.total_irradiance.r:.5f}, {item.total_irradiance.g:.5f}, {item.total_irradiance.b:.5f})",
                f"({item.total_brightness.r:.5f}, {item.total_brightness.g:.5f}, {item.total_brightness.b:.5f})",
                lights_text,
            ]
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(value))
        table.resizeColumnsToContents()

    def _build_details_text(self, result: BrightnessResult) -> str:
        lines = ["Подробный отчет по яркости", ""]
        for scope_name, evaluations in (
            ("Точки, заданные локальными координатами", result.local_evaluations),
            ("Точки, заданные глобальными координатами", result.global_evaluations),
        ):
            lines.append(scope_name)
            lines.append("-" * len(scope_name))
            for item in evaluations:
                lines.append(
                    f"{item.point_name}: P={item.global_point.to_tuple()} "
                    f"E={item.total_irradiance.to_tuple()} L={item.total_brightness.to_tuple()}"
                )
                for contribution in item.contributions:
                    lines.append(
                        f"  {contribution.light_name}: расстояние={contribution.distance:.4f}, "
                        f"theta={contribution.theta_degrees:.2f}°, alpha={contribution.alpha_degrees:.2f}°, "
                        f"BRDF={contribution.brdf_value:.5f}, "
                        f"E={contribution.colored_irradiance.to_tuple()}, "
                        f"L={contribution.brightness.to_tuple()}"
                    )
                lines.append("")
        return "\n".join(lines)

    def _build_help_text(self) -> str:
        return (
            "Что показывает сцена\n"
            "\n"
            "1. Голубая полупрозрачная фигура — сам треугольник в 3D.\n"
            "2. Звезды — точечные источники света.\n"
            "3. Желтые стрелки у источников — направление оси источника.\n"
            "4. Розовый квадрат — положение наблюдателя.\n"
            "5. Цветные точки на треугольнике — точки расчета. Их цвет зависит от рассчитанной яркости.\n"
            "6. Зеленые стрелки — нормали поверхности в точках.\n"
            "7. Пунктирные линии — направления от точек к источникам.\n"
            "8. На вкладке 'Рендер наблюдателя' показывается 2D-изображение поверхности из позиции наблюдателя.\n"
            "9. В таблице 'Глобальные координаты' показывается автоматический пересчет локальных координат в мировые.\n"
            "\n"
            "Что именно считается\n"
            "\n"
            "- По локальным координатам (u, v) точка переносится в глобальные координаты.\n"
            "- По трем вершинам вычисляется нормаль треугольника.\n"
            "- Для каждого источника вычисляются угол к оси источника, диаграмма излучения, расстояние до точки и угол к нормали.\n"
            "- Затем считается цветная освещенность E.\n"
            "- После этого через BRDF считается отраженная яркость L с учетом материала, наблюдателя и полувектора.\n"
            "\n"
            "Важно\n"
            "\n"
            "Треугольник на сцене показан полупрозрачным только для удобства визуализации. "
            "В расчете он не прозрачный: он отражает падающий свет по заданной BRDF-модели. "
            "Это отражение дополнительно визуализируется в 2D-рендере."
        )

    def _build_scene_summary(self, result: BrightnessResult) -> str:
        if not result.local_evaluations:
            return "После расчета здесь появится краткое объяснение сцены и результата."
        normal = result.local_evaluations[0].normal
        brightest = max(
            result.local_evaluations,
            key=lambda item: item.total_brightness.r + item.total_brightness.g + item.total_brightness.b,
        )
        return (
            "Кратко по сцене: нормаль треугольника = "
            f"({normal.x:.3f}, {normal.y:.3f}, {normal.z:.3f}). "
            f"Самая яркая точка сейчас: {brightest.point_name}, "
            f"L = ({brightest.total_brightness.r:.5f}, {brightest.total_brightness.g:.5f}, {brightest.total_brightness.b:.5f}). "
            "Полупрозрачность треугольника нужна только для наглядности в 3D."
        )

    def _clear_results(self) -> None:
        self.local_results_table.setRowCount(0)
        self.global_results_table.setRowCount(0)
        self.details_output.clear()
        self.render_view.clear_render()
        self.help_output.setPlainText(self._build_help_text())
        self.scene_summary.setText("После расчета здесь появится краткое объяснение сцены и результата.")

    def _read_point(self, edits: list[QLineEdit]) -> Point3:
        return Point3(*(self._read_float(edit) for edit in edits))

    def _set_point(self, edits: list[QLineEdit], point: Union[Point3, Vector3]) -> None:
        for edit, value in zip(edits, (point.x, point.y, point.z)):
            edit.setText(str(value))

    def _set_color(self, edits: list[QLineEdit], color: ColorRGB) -> None:
        for edit, value in zip(edits, color.to_tuple()):
            edit.setText(str(value))

    def _read_float(self, widget: QLineEdit) -> float:
        return float(widget.text().strip())

    def _table_text(self, table: QTableWidget, row: int, col: int) -> str:
        item = table.item(row, col)
        if item is None or not item.text().strip():
            raise ValueError(f"Missing table value at row {row + 1}, column {col + 1}.")
        return item.text().strip()

    def _table_float(self, table: QTableWidget, row: int, col: int) -> float:
        return float(self._table_text(table, row, col))

    def _refresh_global_points_preview(self) -> None:
        if self._updating_form:
            return
        try:
            triangle = Triangle(
                a=self._read_point(self.triangle_inputs["a"]),
                b=self._read_point(self.triangle_inputs["b"]),
                c=self._read_point(self.triangle_inputs["c"]),
            )
            local_points = [
                LocalPoint(
                    name=self._table_text(self.local_points_table, row, 0),
                    u=self._table_float(self.local_points_table, row, 1),
                    v=self._table_float(self.local_points_table, row, 2),
                )
                for row in range(self.local_points_table.rowCount())
            ]
        except Exception:
            return

        self._updating_form = True
        self.global_points_table.setRowCount(len(local_points))
        for row, point in enumerate(local_points):
            try:
                global_point = local_to_global(triangle, point.u, point.v)
            except Exception:
                values = (point.name, "Ошибка", "Ошибка", "Ошибка")
                for col, value in enumerate(values):
                    self.global_points_table.setItem(row, col, QTableWidgetItem(str(value)))
                continue
            values = (point.name, global_point.x, global_point.y, global_point.z)
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value if col == 0 else round(value, 6)))
                self.global_points_table.setItem(row, col, item)
        self._updating_form = False

    def _demo_scenario(self) -> Scenario:
        return load_scenario(Path(__file__).resolve().parents[3] / "examples" / "demo_scenario.json")

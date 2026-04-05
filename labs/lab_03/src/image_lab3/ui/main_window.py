from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from image_lab3.io.config_loader import load_config
from image_lab3.models.config import CircleConfig, ExperimentConfig, TriangleConfig
from image_lab3.models.vector import Point3, Vector3
from image_lab3.report.exporters import export_json, export_markdown
from image_lab3.services.experiment_runner import ExperimentRunner
from image_lab3.ui.distribution_canvas import DistributionCanvas


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.runner = ExperimentRunner()
        self.current_config = self._load_demo_config()
        self.current_result = None
        self._canvases = {}
        self.setWindowTitle("ЛР 3 - Формирование распределений случайных величин")
        self.resize(1660, 1020)
        self._build_ui()
        self._load_config_into_form(self.current_config)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        controls = QHBoxLayout()
        self.demo_button = QPushButton("Демо")
        self.load_button = QPushButton("Загрузить конфиг")
        self.run_button = QPushButton("Сгенерировать")
        self.export_button = QPushButton("Экспорт")
        for button in (self.demo_button, self.load_button, self.run_button, self.export_button):
            button.setMinimumHeight(36)
            controls.addWidget(button)
        left_layout.addLayout(controls)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        self.sample_count_input = QLineEdit()
        self.seed_input = QLineEdit()
        self.triangle_a_input = QLineEdit()
        self.triangle_b_input = QLineEdit()
        self.triangle_c_input = QLineEdit()
        self.circle_center_input = QLineEdit()
        self.circle_normal_input = QLineEdit()
        self.circle_radius_input = QLineEdit()
        self.cosine_normal_input = QLineEdit()
        self.uniformity_rectangles_input = QLineEdit()
        form.addRow("Число выборок", self.sample_count_input)
        form.addRow("Seed", self.seed_input)
        form.addRow("Треугольник A", self.triangle_a_input)
        form.addRow("Треугольник B", self.triangle_b_input)
        form.addRow("Треугольник C", self.triangle_c_input)
        form.addRow("Центр круга", self.circle_center_input)
        form.addRow("Нормаль круга", self.circle_normal_input)
        form.addRow("Радиус круга", self.circle_radius_input)
        form.addRow("Нормаль cos-распр.", self.cosine_normal_input)
        form.addRow("Прямоугольники для проверки", self.uniformity_rectangles_input)
        left_layout.addWidget(form_widget)

        self.hint_label = QLabel(
            "ЛР3 должна не только сгенерировать точки и направления, но и доказать корректность распределений. "
            "Поэтому в интерфейсе есть и 3D-визуализация, и проверочные метрики, и гистограммы."
        )
        self.hint_label.setWordWrap(True)
        left_layout.addWidget(self.hint_label)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.tabs = QTabWidget()
        for key, title in (
            ("triangle", "Треугольник"),
            ("circle", "Круг"),
            ("sphere", "Сфера"),
            ("cosine", "Косинусное"),
        ):
            canvas = DistributionCanvas()
            self._canvases[key] = canvas
            self.tabs.addTab(canvas, title)
        right_layout.addWidget(self.summary, stretch=1)
        right_layout.addWidget(self.tabs, stretch=4)

        splitter.addWidget(self._wrap_scroll(left))
        splitter.addWidget(self._wrap_scroll(right))
        splitter.setSizes([470, 1190])

        self.demo_button.clicked.connect(self._on_demo)
        self.load_button.clicked.connect(self._on_load)
        self.run_button.clicked.connect(self._on_run)
        self.export_button.clicked.connect(self._on_export)

        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b1118; color: #e9f1fb; font-size: 13px; }
            QLineEdit, QTextEdit, QTabWidget::pane {
                background: #132131;
                border: 1px solid #22425e;
                border-radius: 10px;
                padding: 6px;
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
            QTabBar::tab {
                background: #122030;
                padding: 8px 14px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
            }
            QTabBar::tab:selected { background: #1d4564; }
            """
        )

    def _on_demo(self) -> None:
        self.current_config = self._load_demo_config()
        self._load_config_into_form(self.current_config)

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить конфиг", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.current_config = load_config(path)
            self._load_config_into_form(self.current_config)
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка загрузки", str(error))

    def _on_run(self) -> None:
        try:
            self.current_config = self._config_from_form()
            self.current_result = self.runner.run(self.current_config)
            self._show_result()
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка расчета", str(error))

    def _on_export(self) -> None:
        if self.current_result is None:
            QMessageBox.information(self, "Экспорт", "Сначала выполните генерацию.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Экспорт результатов")
        if not directory:
            return
        export_dir = Path(directory)
        export_dir.mkdir(parents=True, exist_ok=True)
        export_json(self.current_result, export_dir / "lab3_results.json")
        export_markdown(self.current_result, export_dir / "lab3_results.md")
        QMessageBox.information(self, "Экспорт", "Файлы успешно сохранены.")

    def _show_result(self) -> None:
        lines = ["Сводка проверки распределений", ""]
        for key, distribution in self.current_result.distributions.items():
            self._canvases[key].plot_distribution(distribution)
            metrics = distribution.metrics
            lines.append(distribution.title)
            lines.append(
                "inside_ratio={0:.6f}, constraint_error={1:.6e}, mean_error={2:.6e}, uniformity_score={3:.6f}".format(
                    metrics.inside_ratio,
                    metrics.norm_error,
                    metrics.centroid_or_mean_error,
                    metrics.uniformity_score,
                )
            )
            lines.append(metrics.note)
            lines.append("")
        self.summary.setPlainText("\n".join(lines))

    def _load_config_into_form(self, config: ExperimentConfig) -> None:
        self.sample_count_input.setText(str(config.sample_count))
        self.seed_input.setText(str(config.seed))
        self.triangle_a_input.setText(self._format_vec(config.triangle.a))
        self.triangle_b_input.setText(self._format_vec(config.triangle.b))
        self.triangle_c_input.setText(self._format_vec(config.triangle.c))
        self.circle_center_input.setText(self._format_vec(config.circle.center))
        self.circle_normal_input.setText(self._format_vec(config.circle.normal))
        self.circle_radius_input.setText(str(config.circle.radius))
        self.cosine_normal_input.setText(self._format_vec(config.cosine_normal))
        self.uniformity_rectangles_input.setText(str(config.uniformity_rectangle_count))

    def _config_from_form(self) -> ExperimentConfig:
        return ExperimentConfig(
            sample_count=int(self.sample_count_input.text().strip()),
            seed=int(self.seed_input.text().strip()),
            triangle=TriangleConfig(
                a=self._parse_point(self.triangle_a_input.text()),
                b=self._parse_point(self.triangle_b_input.text()),
                c=self._parse_point(self.triangle_c_input.text()),
            ),
            circle=CircleConfig(
                center=self._parse_point(self.circle_center_input.text()),
                normal=self._parse_vector(self.circle_normal_input.text()),
                radius=float(self.circle_radius_input.text().strip()),
            ),
            cosine_normal=self._parse_vector(self.cosine_normal_input.text()),
            uniformity_rectangle_count=int(self.uniformity_rectangles_input.text().strip()),
        )

    def _parse_point(self, text: str) -> Point3:
        values = [float(item.strip()) for item in text.split(",")]
        return Point3(values[0], values[1], values[2])

    def _parse_vector(self, text: str) -> Vector3:
        values = [float(item.strip()) for item in text.split(",")]
        return Vector3(values[0], values[1], values[2])

    def _format_vec(self, vec) -> str:
        return "{0}, {1}, {2}".format(vec.x, vec.y, vec.z)

    def _wrap_scroll(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    def _load_demo_config(self) -> ExperimentConfig:
        return load_config(Path(__file__).resolve().parents[3] / "examples" / "default_config.json")

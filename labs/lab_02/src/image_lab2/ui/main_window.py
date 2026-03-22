from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from image_lab2.io.config_loader import load_config
from image_lab2.models.config import ExperimentConfig
from image_lab2.models.results import ExperimentResult
from image_lab2.report.exporters import export_csv, export_json, export_markdown
from image_lab2.services.experiment_runner import ExperimentRunner
from image_lab2.ui.charts import ChartsCanvas


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.runner = ExperimentRunner()
        self.current_config = self._load_demo_config()
        self.current_result = None  # type: Optional[ExperimentResult]
        self.setWindowTitle("ЛР 2 - Интегрирование методом Монте-Карло")
        self.resize(1560, 980)
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
        self.run_button = QPushButton("Запустить расчет")
        self.export_button = QPushButton("Экспорт")
        for button in (self.demo_button, self.load_button, self.run_button, self.export_button):
            button.setMinimumHeight(36)
            controls.addWidget(button)
        left_layout.addLayout(controls)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        self.integrand_input = QLineEdit()
        self.interval_left_input = QLineEdit()
        self.interval_right_input = QLineEdit()
        self.sample_sizes_input = QLineEdit()
        self.strat_steps_input = QLineEdit()
        self.is_powers_input = QLineEdit()
        self.mis_powers_input = QLineEdit()
        self.rr_thresholds_input = QLineEdit()
        self.seed_input = QLineEdit()
        form.addRow("Функция", self.integrand_input)
        form.addRow("Левая граница", self.interval_left_input)
        form.addRow("Правая граница", self.interval_right_input)
        form.addRow("Размеры выборки", self.sample_sizes_input)
        form.addRow("Шаги стратификации", self.strat_steps_input)
        form.addRow("Степени p(x) для IS", self.is_powers_input)
        form.addRow("Степени p(x) для MIS", self.mis_powers_input)
        form.addRow("Пороги русской рулетки", self.rr_thresholds_input)
        form.addRow("Seed", self.seed_input)
        left_layout.addWidget(form_widget)

        self.summary_label = QLabel(
            "ЛР2 использует f(x)=x^2 на [2,5] и сравнивает аналитический интеграл "
            "с несколькими вариантами метода Монте-Карло."
        )
        self.summary_label.setWordWrap(True)
        left_layout.addWidget(self.summary_label)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.charts = ChartsCanvas()
        self.results_table = QTableWidget(0, 7)
        self.results_table.setHorizontalHeaderLabels(
            ["Метод", "N", "Оценка", "Истинное", "Абс. ошибка", "Отн. ошибка", "Оценка погрешности"]
        )
        self.results_table.setSortingEnabled(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMinimumHeight(320)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(150)
        right_layout.addWidget(self.charts, stretch=3)
        right_layout.addWidget(self.results_table, stretch=2)
        right_layout.addWidget(self.details, stretch=1)

        left_scroll = self._wrap_scroll(left)
        right_scroll = self._wrap_scroll(right)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_scroll)
        splitter.setSizes([460, 1100])

        self.demo_button.clicked.connect(self._on_demo)
        self.load_button.clicked.connect(self._on_load)
        self.run_button.clicked.connect(self._on_run)
        self.export_button.clicked.connect(self._on_export)

        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b1118; color: #e9f1fb; font-size: 13px; }
            QLineEdit, QTextEdit, QTableWidget {
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
            QHeaderView::section {
                background: #16324b;
                color: #f3f8ff;
                border: none;
                padding: 6px;
            }
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
            self._show_result(self.current_result)
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка расчета", str(error))

    def _on_export(self) -> None:
        if self.current_result is None:
            QMessageBox.information(self, "Экспорт", "Сначала выполните расчет.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Экспорт результатов")
        if not directory:
            return
        export_dir = Path(directory)
        export_dir.mkdir(parents=True, exist_ok=True)
        export_json(self.current_result, export_dir / "lab2_results.json")
        export_csv(self.current_result, export_dir / "lab2_results.csv")
        export_markdown(self.current_result, export_dir / "lab2_results.md")
        QMessageBox.information(self, "Экспорт", "Файлы успешно сохранены.")

    def _show_result(self, result: ExperimentResult) -> None:
        rows = sum(len(series.estimates) for series in result.by_method.values())
        self.results_table.setSortingEnabled(False)
        self.results_table.setRowCount(rows)
        row = 0
        for series in result.by_method.values():
            for estimate in series.estimates:
                values = [
                    estimate.display_name,
                    str(estimate.sample_size),
                    "{0:.8f}".format(estimate.estimate),
                    "{0:.8f}".format(estimate.true_value),
                    "{0:.8f}".format(estimate.absolute_error),
                    "{0:.8f}".format(estimate.relative_error),
                    "{0:.8f}".format(estimate.estimated_error),
                ]
                for column, value in enumerate(values):
                    self.results_table.setItem(row, column, QTableWidgetItem(value))
                row += 1
        self.results_table.resizeColumnsToContents()
        self.results_table.setSortingEnabled(True)
        self.charts.plot_result(result)
        self.details.setPlainText(self._build_details_text(result))

    def _build_details_text(self, result: ExperimentResult) -> str:
        best = min(
            (estimate for series in result.by_method.values() for estimate in series.estimates),
            key=lambda item: item.absolute_error,
        )
        note = ""
        if self.current_config.interval.left < 0:
            note = (
                "\n\nПримечание: для отрицательных интервалов методы значимости и MIS "
                "автоматически используют сдвинутые плотности вида p(x)~(x-a)^k, "
                "чтобы вычисления оставались корректными и не давали NaN."
            )
        return (
            "Истинный интеграл: {0:.8f}\n"
            "Формула оценки погрешности: {1}\n\n"
            "Лучшая оценка по абсолютной ошибке:\n"
            "{2}, N={3}, I={4:.8f}, |Δ|={5:.8f}{6}".format(
                result.true_value,
                result.estimated_error_formula,
                best.display_name,
                best.sample_size,
                best.estimate,
                best.absolute_error,
                note,
            )
        )

    def _config_from_form(self) -> ExperimentConfig:
        return ExperimentConfig(
            integrand=self.integrand_input.text().strip(),
            interval=self.current_config.interval.__class__(
                left=float(self.interval_left_input.text().strip()),
                right=float(self.interval_right_input.text().strip()),
            ),
            sample_sizes=self._parse_int_list(self.sample_sizes_input.text()),
            stratification_steps=self._parse_float_list(self.strat_steps_input.text()),
            importance_sampling_powers=self._parse_int_list(self.is_powers_input.text()),
            mis_powers=self._parse_int_list(self.mis_powers_input.text()),
            russian_roulette_thresholds=self._parse_float_list(self.rr_thresholds_input.text()),
            seed=int(self.seed_input.text().strip()),
        )

    def _load_config_into_form(self, config: ExperimentConfig) -> None:
        self.integrand_input.setText(config.integrand)
        self.interval_left_input.setText(str(config.interval.left))
        self.interval_right_input.setText(str(config.interval.right))
        self.sample_sizes_input.setText(", ".join(str(value) for value in config.sample_sizes))
        self.strat_steps_input.setText(", ".join(str(value) for value in config.stratification_steps))
        self.is_powers_input.setText(", ".join(str(value) for value in config.importance_sampling_powers))
        self.mis_powers_input.setText(", ".join(str(value) for value in config.mis_powers))
        self.rr_thresholds_input.setText(", ".join(str(value) for value in config.russian_roulette_thresholds))
        self.seed_input.setText(str(config.seed))

    def _parse_int_list(self, value: str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]

    def _parse_float_list(self, value: str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]

    def _wrap_scroll(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    def _load_demo_config(self) -> ExperimentConfig:
        return load_config(Path(__file__).resolve().parents[3] / "examples" / "default_config.json")

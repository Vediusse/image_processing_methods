from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
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
from image_lab2.models.config import ExperimentConfig, Interval
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
        self.setWindowTitle("ЛР 2 - Monte Carlo Variance Studio")
        self.resize(1680, 1020)
        self._build_ui()
        self._load_config_into_form(self.current_config)
        self._reset_dashboard()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        root.addWidget(self._build_hero())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._wrap_scroll(self._build_control_panel()))
        splitter.addWidget(self._wrap_scroll(self._build_results_panel()))
        splitter.setSizes([470, 1170])
        root.addWidget(splitter, stretch=1)

        self._apply_style()

    def _build_hero(self) -> QFrame:
        hero = QFrame()
        hero.setObjectName("hero")
        layout = QHBoxLayout(hero)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(6)

        eyebrow = QLabel("Лабораторная работа №2")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Monte Carlo Variance Studio")
        title.setObjectName("heroTitle")
        subtitle = QLabel(
            "Новый сценарный интерфейс для сравнения simple MC, адаптивной стратификации, "
            "importance sampling, MIS и русской рулетки на одной панели."
        )
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)
        text_layout.addWidget(eyebrow)
        text_layout.addWidget(title)
        text_layout.addWidget(subtitle)
        text_layout.addStretch(1)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.demo_button = QPushButton("Сбросить демо")
        self.load_button = QPushButton("Открыть JSON")
        self.run_button = QPushButton("Пересчитать")
        self.export_button = QPushButton("Экспортировать")
        self.run_button.setObjectName("primaryButton")
        self.export_button.setObjectName("accentButton")
        for button in (self.demo_button, self.load_button, self.run_button, self.export_button):
            button.setMinimumHeight(42)
            actions.addWidget(button)
        text_layout.addLayout(actions)
        layout.addLayout(text_layout, stretch=5)

        info_card = QFrame()
        info_card.setObjectName("heroSideCard")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(18, 18, 18, 18)
        info_layout.setSpacing(8)
        caption = QLabel("Что изменено")
        caption.setObjectName("sideCardTitle")
        blurb = QLabel(
            "Стратификация теперь использует аналитическую локальную дисперсию и распределение "
            "точек по Нейману, поэтому уменьшение шага действительно снижает разброс оценки."
        )
        blurb.setObjectName("sideCardText")
        blurb.setWordWrap(True)
        info_layout.addWidget(caption)
        info_layout.addWidget(blurb)
        info_layout.addStretch(1)
        layout.addWidget(info_card, stretch=2)

        self.demo_button.clicked.connect(self._on_demo)
        self.load_button.clicked.connect(self._on_load)
        self.run_button.clicked.connect(self._on_run)
        self.export_button.clicked.connect(self._on_export)
        return hero

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(14)

        config_card = self._create_card("Сценарий эксперимента", "Заполни параметры или загрузи конфиг.")
        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(24, 58, 24, 24)
        config_layout.setSpacing(14)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
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
        config_layout.addWidget(form_widget)

        self.summary_label = QLabel()
        self.summary_label.setObjectName("noteText")
        self.summary_label.setWordWrap(True)
        config_layout.addWidget(self.summary_label)

        blueprint_card = self._create_card("Дизайн эксперимента", "Краткая шпаргалка по сценариям.")
        blueprint_layout = QVBoxLayout(blueprint_card)
        blueprint_layout.setContentsMargins(24, 58, 24, 24)
        blueprint_layout.setSpacing(10)
        self.blueprint_label = QLabel(
            "Simple MC задаёт базовую линию. Адаптивная стратификация перераспределяет точки "
            "туда, где локальная дисперсия выше. IS и MIS проверяют, насколько хорошо плотности "
            "совпадают с формой f(x)=x^2."
        )
        self.blueprint_label.setObjectName("noteText")
        self.blueprint_label.setWordWrap(True)
        blueprint_layout.addWidget(self.blueprint_label)

        layout.addWidget(config_card)
        layout.addWidget(blueprint_card)
        layout.addStretch(1)
        return panel

    def _build_results_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(14)

        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(14)
        self.true_value_card = self._create_metric_card("Истинный интеграл", "Ожидание")
        self.best_method_card = self._create_metric_card("Лучший метод", "Лидер по |Δ|")
        self.best_error_card = self._create_metric_card("Лучшая ошибка", "Минимум")
        self.strat_gain_card = self._create_metric_card("Выигрыш стратификации", "vs simple MC")
        metrics_grid.addWidget(self.true_value_card, 0, 0)
        metrics_grid.addWidget(self.best_method_card, 0, 1)
        metrics_grid.addWidget(self.best_error_card, 1, 0)
        metrics_grid.addWidget(self.strat_gain_card, 1, 1)
        layout.addLayout(metrics_grid)

        chart_card = self._create_card("Аналитическая панель", "Сходимость, фактическая ошибка и оценка stderr.")
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(20, 58, 20, 20)
        self.charts = ChartsCanvas()
        chart_layout.addWidget(self.charts)
        layout.addWidget(chart_card, stretch=3)

        lower_splitter = QSplitter(Qt.Vertical)
        lower_splitter.addWidget(self._build_table_card())
        lower_splitter.addWidget(self._build_details_card())
        lower_splitter.setSizes([420, 220])
        layout.addWidget(lower_splitter, stretch=2)
        return panel

    def _build_table_card(self) -> QFrame:
        table_card = self._create_card("Результаты", "Таблица всех запусков с сортировкой.")
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(20, 58, 20, 20)
        self.results_table = QTableWidget(0, 7)
        self.results_table.setHorizontalHeaderLabels(
            ["Метод", "N", "Оценка", "Истинное", "|Δ|", "Δ / I", "stderr"]
        )
        self.results_table.setSortingEnabled(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.results_table)
        return table_card

    def _build_details_card(self) -> QFrame:
        details_card = self._create_card("Диагностика", "Краткий разбор результата текущего прогона.")
        details_layout = QVBoxLayout(details_card)
        details_layout.setContentsMargins(20, 58, 20, 20)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        details_layout.addWidget(self.details)
        return details_card

    def _create_card(self, title: str, subtitle: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")

        title_label = QLabel(title, card)
        title_label.setObjectName("cardTitle")
        title_label.move(24, 18)

        subtitle_label = QLabel(subtitle, card)
        subtitle_label.setObjectName("cardSubtitle")
        subtitle_label.move(24, 38)
        return card

    def _create_metric_card(self, title: str, caption: str) -> QFrame:
        card = QFrame()
        card.setObjectName("metricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label = QLabel("—")
        value_label.setObjectName("metricValue")
        caption_label = QLabel(caption)
        caption_label.setObjectName("metricCaption")
        caption_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(caption_label)
        card.value_label = value_label  # type: ignore[attr-defined]
        card.caption_label = caption_label  # type: ignore[attr-defined]
        return card

    def _set_metric(self, card: QFrame, value: str, caption: str) -> None:
        card.value_label.setText(value)  # type: ignore[attr-defined]
        card.caption_label.setText(caption)  # type: ignore[attr-defined]

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f3ede2;
                color: #17323a;
                font-size: 13px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QFrame#hero {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #12343d, stop: 0.55 #27545d, stop: 1 #d46d4f);
                border-radius: 28px;
            }
            QLabel#eyebrow {
                color: #f7d4bf;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 1px;
                text-transform: uppercase;
            }
            QLabel#heroTitle {
                color: #fff7ef;
                font-size: 30px;
                font-weight: 800;
            }
            QLabel#heroSubtitle, QLabel#sideCardText {
                color: #e7efe8;
                font-size: 14px;
                line-height: 1.3em;
            }
            QFrame#heroSideCard {
                background: rgba(255, 250, 243, 0.12);
                border: 1px solid rgba(255, 241, 229, 0.35);
                border-radius: 22px;
            }
            QLabel#sideCardTitle {
                color: #fff7ef;
                font-size: 16px;
                font-weight: 700;
            }
            QFrame#card, QFrame#metricCard {
                background: #fffaf2;
                border: 1px solid #ddcfba;
                border-radius: 22px;
            }
            QLabel#cardTitle {
                color: #17323a;
                font-size: 18px;
                font-weight: 700;
                background: transparent;
            }
            QLabel#cardSubtitle {
                color: #7b6d60;
                font-size: 12px;
                background: transparent;
            }
            QLabel#metricTitle {
                color: #7b6d60;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
            }
            QLabel#metricValue {
                color: #17323a;
                font-size: 24px;
                font-weight: 800;
            }
            QLabel#metricCaption, QLabel#noteText {
                color: #5b544c;
                font-size: 13px;
                line-height: 1.35em;
            }
            QLineEdit, QTextEdit, QTableWidget {
                background: #fffdf8;
                border: 1px solid #d8c9b0;
                border-radius: 16px;
                padding: 8px 10px;
                selection-background-color: #17323a;
                selection-color: #fff7ef;
            }
            QLineEdit:focus, QTextEdit:focus, QTableWidget:focus {
                border: 1px solid #d46d4f;
            }
            QPushButton {
                background: #f9f3ea;
                border: 1px solid rgba(255, 250, 243, 0.55);
                border-radius: 14px;
                padding: 10px 16px;
                font-weight: 700;
                color: #17323a;
            }
            QPushButton:hover {
                background: #fff7ef;
            }
            QPushButton#primaryButton {
                background: #fff3e7;
                color: #17323a;
                border: 1px solid #f7d5bc;
            }
            QPushButton#accentButton {
                background: #d46d4f;
                color: #fff7ef;
                border: 1px solid #d46d4f;
            }
            QPushButton#accentButton:hover {
                background: #bf5b3e;
            }
            QHeaderView::section {
                background: #efe5d5;
                color: #17323a;
                border: none;
                border-bottom: 1px solid #d8c9b0;
                padding: 8px;
                font-weight: 700;
            }
            QTableWidget {
                gridline-color: #ebdfcf;
            }
            QSplitter::handle {
                background: #e4d8c5;
                border-radius: 2px;
            }
            """
        )

    def _on_demo(self) -> None:
        self.current_config = self._load_demo_config()
        self._load_config_into_form(self.current_config)
        self._reset_dashboard()

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить конфиг", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.current_config = load_config(path)
            self._load_config_into_form(self.current_config)
            self._reset_dashboard()
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
        self._update_metrics(result)

    def _update_metrics(self, result: ExperimentResult) -> None:
        estimates = [estimate for series in result.by_method.values() for estimate in series.estimates]
        best = min(estimates, key=lambda item: item.absolute_error)
        self._set_metric(self.true_value_card, "{0:.8f}".format(result.true_value), "Аналитическое значение интеграла.")
        self._set_metric(
            self.best_method_card,
            best.display_name,
            "При N={0} получено минимальное |Δ|.".format(best.sample_size),
        )
        self._set_metric(
            self.best_error_card,
            "{0:.8f}".format(best.absolute_error),
            "Оценка {0:.8f}".format(best.estimate),
        )

        gain_value, gain_caption = self._build_stratification_gain(result)
        self._set_metric(self.strat_gain_card, gain_value, gain_caption)

    def _build_stratification_gain(self, result: ExperimentResult) -> Tuple[str, str]:
        simple = result.by_method.get("simple_mc")
        stratified = [series for key, series in result.by_method.items() if key.startswith("stratified_")]
        if simple is None or not stratified:
            return "—", "Недостаточно данных для сравнения."
        simple_map = {estimate.sample_size: estimate for estimate in simple.estimates}
        comparisons = []
        for series in stratified:
            for estimate in series.estimates:
                baseline = simple_map.get(estimate.sample_size)
                if baseline is None or estimate.absolute_error <= 0.0:
                    continue
                comparisons.append((baseline.absolute_error / estimate.absolute_error, series.display_name, estimate.sample_size))
        if not comparisons:
            return "—", "Для текущего запуска выигрыш не вычислен."
        best_gain, method_name, sample_size = max(comparisons, key=lambda item: item[0])
        return "{0:.2f}x".format(best_gain), "{0} при N={1}".format(method_name, sample_size)

    def _build_details_text(self, result: ExperimentResult) -> str:
        estimates = [estimate for series in result.by_method.values() for estimate in series.estimates]
        best = min(estimates, key=lambda item: item.absolute_error)
        largest_n = max(estimate.sample_size for estimate in estimates)
        final_best = min((estimate for estimate in estimates if estimate.sample_size == largest_n), key=lambda item: item.absolute_error)

        note = ""
        if self.current_config.interval.left < 0:
            note = (
                "\n\nПримечание: на отрицательных интервалах importance sampling и MIS "
                "используют сдвинутые плотности p(x)~(x-a)^k, чтобы избежать вырожденных значений."
            )

        gain_value, gain_caption = self._build_stratification_gain(result)
        return (
            "Истинный интеграл: {0:.8f}\n"
            "Формула оценки погрешности: {1}\n\n"
            "Глобальный лидер:\n"
            "{2}, N={3}, I={4:.8f}, |Δ|={5:.8f}\n\n"
            "Лучший метод на максимальном размере выборки:\n"
            "{6}, N={7}, I={8:.8f}, |Δ|={9:.8f}, stderr={10:.8f}\n\n"
            "Диагностика стратификации:\n"
            "Максимальный выигрыш относительно simple MC: {11}\n"
            "{12}{13}".format(
                result.true_value,
                result.estimated_error_formula,
                best.display_name,
                best.sample_size,
                best.estimate,
                best.absolute_error,
                final_best.display_name,
                final_best.sample_size,
                final_best.estimate,
                final_best.absolute_error,
                final_best.estimated_error,
                gain_value,
                gain_caption,
                note,
            )
        )

    def _reset_dashboard(self) -> None:
        self.results_table.setRowCount(0)
        self.details.setPlainText(
            "После запуска здесь появится краткий разбор результата: лучший метод, ошибка и выигрыш адаптивной стратификации."
        )
        self._set_metric(self.true_value_card, "—", "Ожидает расчёта.")
        self._set_metric(self.best_method_card, "—", "Лидер появится после прогона.")
        self._set_metric(self.best_error_card, "—", "Пока нет численной оценки.")
        self._set_metric(self.strat_gain_card, "—", "Сравнение с simple MC появится после запуска.")

    def _config_from_form(self) -> ExperimentConfig:
        return ExperimentConfig(
            integrand=self.integrand_input.text().strip(),
            interval=Interval(
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
        self.summary_label.setText(
            "Сценарий: f(x)=x^2 на [{0:.2f}, {1:.2f}] с N = {2}. "
            "Стратификация теперь использует адаптивное распределение точек по локальной дисперсии, "
            "поэтому уменьшение шага должно уменьшать ошибку стабильнее, чем раньше.".format(
                config.interval.left,
                config.interval.right,
                ", ".join(str(value) for value in config.sample_sizes),
            )
        )

    def _load_demo_config(self) -> ExperimentConfig:
        return load_config(Path(__file__).resolve().parents[3] / "examples" / "default_config.json")

    def _parse_int_list(self, raw_value: str):
        return [int(item.strip()) for item in raw_value.split(",") if item.strip()]

    def _parse_float_list(self, raw_value: str):
        return [float(item.strip()) for item in raw_value.split(",") if item.strip()]

    def _wrap_scroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

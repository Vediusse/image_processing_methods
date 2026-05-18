from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from image_lab4.io.config_loader import load_config, load_config_from_text
from image_lab4.models.scene import DenoiseSettings, RenderArtifact
from image_lab4.report.exporters import save_hdr, save_png, save_ppm
from image_lab4.services.image_filters import FilterGuide, FilterSettings, ImageFilterService, split_bilateral_denoise
from image_lab4.services.path_tracer import PathTracer
from image_lab4.services.taichi_progressive_path_tracer import TaichiProgressivePathTracer, _build_gbuffer, _tone_map


def _comparison_image(raw: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    filtered = np.asarray(filtered, dtype=np.float32)
    height, width, _ = raw.shape
    separator_width = max(2, width // 160)
    separator = np.full((height, separator_width, 3), 255.0, dtype=np.float32)
    return np.concatenate([raw, separator, filtered], axis=1)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.current_path = Path(__file__).resolve().parents[3] / "examples" / "default_scene.json"
        self.current_artifact = None
        self.tracer = PathTracer()
        self.gpu_path_tracer = TaichiProgressivePathTracer()
        self.realtime_timer = QTimer(self)
        self.realtime_timer.setInterval(33)
        self.realtime_timer.timeout.connect(self._render_realtime_frame)
        self.realtime_config = None
        self.realtime_config_text = ""
        self.realtime_state = None
        self.realtime_accumulation = None
        self.realtime_direct_accumulation = None
        self.current_raw_radiance = None
        self.current_direct_radiance = None
        self.current_raw_display = None
        self.realtime_frame_count = 0
        self.config_panel_hidden = False
        self.setWindowTitle("ЛР 4 - Трассировка путей")
        self.resize(1600, 980)
        self._build_ui()
        self._load_demo()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.splitter = QSplitter()
        root.addWidget(self.splitter)

        left = QWidget()
        self.config_panel = left
        left_layout = QVBoxLayout(left)
        controls = QHBoxLayout()
        self.demo_button = QPushButton("Демо")
        self.load_button = QPushButton("Открыть JSON")
        controls.addWidget(self.demo_button)
        controls.addWidget(self.load_button)
        left_layout.addLayout(controls)
        self.config_editor = QTextEdit()
        self.config_editor.setObjectName("configEditor")
        left_layout.addWidget(self.config_editor, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        render_controls = QHBoxLayout()
        self.toggle_config_button = QPushButton("JSON скрыть")
        self.realtime_button = QPushButton("GPU PathTrace старт")
        self.preview_button = QPushButton("Быстрый превью")
        self.render_button = QPushButton("GPU финал до SPP")
        self.save_ppm_button = QPushButton("Сохранить PPM")
        self.save_png_button = QPushButton("Сохранить PNG")
        self.save_hdr_button = QPushButton("Сохранить HDR")
        for button in (
            self.toggle_config_button,
            self.realtime_button,
            self.preview_button,
            self.render_button,
            self.save_ppm_button,
            self.save_png_button,
            self.save_hdr_button,
        ):
            render_controls.addWidget(button)
        right_layout.addLayout(render_controls)

        filter_controls = QHBoxLayout()
        self.denoise_enabled = QCheckBox("Фильтр")
        self.denoise_enabled.setChecked(True)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(sorted(ImageFilterService().names))
        self.filter_combo.setCurrentText("bilateral")
        self.filter_radius = QSpinBox()
        self.filter_radius.setRange(1, 8)
        self.filter_radius.setValue(2)
        self.filter_strength = QDoubleSpinBox()
        self.filter_strength.setRange(0.0, 1.0)
        self.filter_strength.setSingleStep(0.05)
        self.filter_strength.setDecimals(2)
        self.filter_strength.setValue(0.95)
        self.compare_checkbox = QCheckBox("До/после")
        self.compare_checkbox.setChecked(True)
        self.denoise_enabled.stateChanged.connect(self._on_filter_controls_changed)
        self.filter_combo.currentTextChanged.connect(self._on_filter_controls_changed)
        self.filter_radius.valueChanged.connect(self._on_filter_controls_changed)
        self.filter_strength.valueChanged.connect(self._on_filter_controls_changed)
        self.compare_checkbox.stateChanged.connect(self._on_filter_controls_changed)
        filter_controls.addWidget(self.denoise_enabled)
        filter_controls.addWidget(QLabel("Тип"))
        filter_controls.addWidget(self.filter_combo)
        filter_controls.addWidget(QLabel("Радиус"))
        filter_controls.addWidget(self.filter_radius)
        filter_controls.addWidget(QLabel("Сила"))
        filter_controls.addWidget(self.filter_strength)
        filter_controls.addWidget(self.compare_checkbox)
        filter_controls.addStretch(1)
        right_layout.addLayout(filter_controls)

        self.preview = QLabel(
            "После рендера здесь появится изображение.\n"
            "GPU PathTrace стартует progressive Monte-Carlo.\n"
            "Камеру вращай стрелками ← / →."
        )
        self.preview.setMinimumSize(360, 360)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("QLabel { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #101925, stop:1 #1a2a39); color: #d9e8ff; border: 1px solid #24364c; border-radius: 18px; }")
        self.preview.setScaledContents(False)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setObjectName("summaryPanel")
        right_layout.addWidget(self.preview, stretch=5)
        right_layout.addWidget(self.summary, stretch=2)

        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.splitter.setCollapsible(0, True)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([420, 1180])

        self.demo_button.clicked.connect(self._load_demo)
        self.load_button.clicked.connect(self._on_open)
        self.toggle_config_button.clicked.connect(self._toggle_config_panel)
        self.realtime_button.clicked.connect(self._toggle_realtime_preview)
        self.preview_button.clicked.connect(lambda: self._on_render(preview=True))
        self.render_button.clicked.connect(self._on_render)
        self.save_ppm_button.clicked.connect(self._on_save_ppm)
        self.save_png_button.clicked.connect(self._on_save_png)
        self.save_hdr_button.clicked.connect(self._on_save_hdr)
        self._add_camera_shortcuts()
        self._apply_style()

    def _toggle_config_panel(self) -> None:
        self.config_panel_hidden = not self.config_panel_hidden
        self.config_panel.setVisible(not self.config_panel_hidden)
        self.toggle_config_button.setText("JSON показать" if self.config_panel_hidden else "JSON скрыть")
        if self.config_panel_hidden:
            self.splitter.setSizes([0, max(1, self.width())])
        else:
            self.splitter.setSizes([420, max(1, self.width() - 420)])

    def _add_camera_shortcuts(self) -> None:
        self.camera_left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.camera_left_shortcut.setContext(Qt.ApplicationShortcut)
        self.camera_left_shortcut.activated.connect(lambda: self._rotate_camera(-10.0))

        self.camera_right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.camera_right_shortcut.setContext(Qt.ApplicationShortcut)
        self.camera_right_shortcut.activated.connect(lambda: self._rotate_camera(10.0))

    def _load_demo(self) -> None:
        self.config_editor.setPlainText(self.current_path.read_text(encoding="utf-8"))
        self._sync_filter_controls_from_json()

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть конфиг", "", "JSON (*.json)")
        if not path:
            return
        self.current_path = Path(path)
        self.config_editor.setPlainText(self.current_path.read_text(encoding="utf-8"))
        self._sync_filter_controls_from_json()

    def _on_render(self, preview: bool = False) -> None:
        try:
            config_text = self.config_editor.toPlainText()
            config = self._apply_filter_controls(load_config_from_text(config_text, base_dir=self.current_path.parent))
            if preview:
                render = config.render.__class__(
                    width=min(256, config.render.width),
                    height=min(256, config.render.height),
                    samples_per_pixel=max(1, min(1, config.render.samples_per_pixel)),
                    max_depth=min(2, config.render.max_depth),
                    min_depth=min(1, config.render.min_depth),
                    gamma=config.render.gamma,
                    normalization=config.render.normalization,
                    normalization_value=config.render.normalization_value,
                    seed=config.render.seed,
                    background=config.render.background,
                )
                config = config.__class__(
                    camera=config.camera,
                    render=render,
                    materials=config.materials,
                    triangles=config.triangles,
                    obj_meshes=config.obj_meshes,
                    point_lights=config.point_lights,
                    denoise=config.denoise,
                )
                self.current_artifact = self.tracer.render(
                    config,
                    config_path=self.current_path,
                    strict_resolution=False,
                )
                self.summary.setPlainText(self.current_artifact.summary)
                self._show_image(self.current_artifact.display)
                return
            self._render_gpu_final(config)
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка рендера", str(error))

    def _render_gpu_final(self, config) -> None:
        was_running = self.realtime_timer.isActive()
        if was_running:
            self.realtime_timer.stop()
            self.realtime_button.setText("GPU PathTrace старт")

        target_frames = max(1, int(config.render.samples_per_pixel))
        config_text = self.config_editor.toPlainText().strip()
        if self._gpu_state_matches_config(config, config_text):
            self.realtime_config = config
            if self.realtime_accumulation is None:
                self.realtime_accumulation = np.zeros(
                    (int(config.render.height), int(config.render.width), 3),
                    dtype=np.float32,
                )
            if self.realtime_direct_accumulation is None:
                self.realtime_direct_accumulation = np.zeros_like(self.realtime_accumulation)
        else:
            self.realtime_config = config
            self.realtime_config_text = config_text
            self.realtime_state = self.gpu_path_tracer.create_state(config, config_path=self.current_path)
            self._reset_gpu_accumulation(clear_config=False)

        start_frame = self.realtime_frame_count
        if start_frame >= target_frames and self.current_artifact is not None:
            self.summary.append("\nТекущий GPU кадр уже достиг SPP={0}; можно сохранять PNG/HDR/PPM.".format(target_frames))
            return

        self.render_button.setEnabled(False)
        self.realtime_button.setEnabled(False)
        try:
            started = perf_counter()
            for frame_index in range(start_frame, target_frames):
                frame_started = perf_counter()
                frame_radiance = self.gpu_path_tracer.trace_state_frame(self.realtime_state, seed_offset=frame_index)
                if self.realtime_accumulation is None:
                    self.realtime_accumulation = np.zeros_like(frame_radiance)
                if self.realtime_direct_accumulation is None:
                    self.realtime_direct_accumulation = np.zeros_like(frame_radiance)
                self.realtime_accumulation += frame_radiance
                self.realtime_direct_accumulation += self.realtime_state["frame_direct"]
                self.realtime_frame_count += 1
                averaged = self.realtime_accumulation / float(self.realtime_frame_count)
                direct = self.realtime_direct_accumulation / float(self.realtime_frame_count)
                self._set_gpu_artifact(averaged, direct)
                if frame_index == start_frame or self.realtime_frame_count == target_frames or self.realtime_frame_count % 4 == 0:
                    fps = 1.0 / max(perf_counter() - frame_started, 1e-9)
                    self._update_gpu_summary(
                        prefix="GPU final progressive path tracing",
                        fps=fps,
                        extra="\nTarget SPP: {0}\nElapsed: {1:.2f} s".format(
                            target_frames,
                            perf_counter() - started,
                        ),
                    )
                    self._show_image(self.current_artifact.display)
                    QApplication.processEvents()
            self._update_gpu_summary(
                prefix="GPU final progressive path tracing",
                fps=float("nan"),
                extra="\nГотово: накоплено SPP={0}. Можно сохранять текущий PNG/HDR/PPM.".format(self.realtime_frame_count),
            )
            self._show_image(self.current_artifact.display)
        finally:
            self.render_button.setEnabled(True)
            self.realtime_button.setEnabled(True)

    def _gpu_state_matches_config(self, config, config_text: str) -> bool:
        if self.realtime_config is None or self.realtime_state is None:
            return False
        return (
            self.realtime_config_text == config_text
            and int(self.realtime_config.render.width) == int(config.render.width)
            and int(self.realtime_config.render.height) == int(config.render.height)
            and int(self.realtime_config.render.max_depth) == int(config.render.max_depth)
            and int(self.realtime_config.render.min_depth) == int(config.render.min_depth)
            and int(self.realtime_config.render.seed) == int(config.render.seed)
        )

    def _toggle_realtime_preview(self) -> None:
        if self.realtime_timer.isActive():
            self.realtime_timer.stop()
            self.realtime_button.setText("GPU PathTrace старт")
            return
        try:
            self.realtime_config_text = self.config_editor.toPlainText().strip()
            config = self._apply_filter_controls(
                load_config_from_text(self.realtime_config_text, base_dir=self.current_path.parent)
            )
            if not self._gpu_state_matches_config(config, self.realtime_config_text):
                self.realtime_config = config
                self.realtime_state = self.gpu_path_tracer.create_state(self.realtime_config, config_path=self.current_path)
                self._reset_gpu_accumulation(clear_config=False)
            else:
                self.realtime_config = config
            target_spp = max(1, int(self.realtime_config.render.samples_per_pixel))
            if self.realtime_frame_count >= target_spp and self.current_artifact is not None:
                self._update_gpu_summary(
                    "GPU progressive path tracing",
                    fps=float("nan"),
                    extra="\nГотово: достигнут target SPP={0}. Измени samples_per_pixel или сцену, чтобы начать новый набор.".format(target_spp),
                )
                return
            self.realtime_button.setText("GPU PathTrace стоп")
            self._render_realtime_frame()
            if self.realtime_frame_count < target_spp:
                self.realtime_timer.start()
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка GPU path tracing", str(error))

    def _render_realtime_frame(self) -> None:
        if self.realtime_config is None:
            return
        try:
            target_spp = max(1, int(self.realtime_config.render.samples_per_pixel))
            if self.realtime_frame_count >= target_spp:
                self.realtime_timer.stop()
                self.realtime_button.setText("GPU PathTrace старт")
                self._update_gpu_summary(
                    "GPU progressive path tracing",
                    fps=float("nan"),
                    extra="\nГотово: накоплено SPP={0}/{1}. Можно сохранять PNG/HDR/PPM.".format(
                        self.realtime_frame_count,
                        target_spp,
                    ),
                )
                return
            started = perf_counter()
            if self.realtime_state is None:
                self.realtime_state = self.gpu_path_tracer.create_state(self.realtime_config, config_path=self.current_path)
            frame_radiance = self.gpu_path_tracer.trace_state_frame(self.realtime_state, seed_offset=self.realtime_frame_count)
            elapsed = max(perf_counter() - started, 1e-9)
            fps = 1.0 / elapsed
            if self.realtime_accumulation is None:
                self.realtime_accumulation = np.zeros_like(frame_radiance)
            if self.realtime_direct_accumulation is None:
                self.realtime_direct_accumulation = np.zeros_like(frame_radiance)
            self.realtime_accumulation += frame_radiance
            self.realtime_direct_accumulation += self.realtime_state["frame_direct"]
            self.realtime_frame_count += 1
            averaged = self.realtime_accumulation / float(self.realtime_frame_count)
            direct = self.realtime_direct_accumulation / float(self.realtime_frame_count)
            self._set_gpu_artifact(averaged, direct)
            extra = "\nTarget SPP: {0}".format(target_spp)
            if self.realtime_frame_count >= target_spp:
                self.realtime_timer.stop()
                self.realtime_button.setText("GPU PathTrace старт")
                extra += "\nГотово: достигнут target SPP."
            self._update_gpu_summary("GPU progressive path tracing", fps=fps, extra=extra)
            self._show_image(self.current_artifact.display)
        except Exception as error:  # noqa: BLE001
            self.realtime_timer.stop()
            self.realtime_button.setText("GPU PathTrace старт")
            QMessageBox.critical(self, "Ошибка GPU path tracing", str(error))

    def _set_gpu_artifact(self, radiance: np.ndarray, direct: Optional[np.ndarray] = None) -> None:
        self.current_raw_radiance = radiance
        self.current_direct_radiance = direct
        self.current_raw_display = _tone_map(
            radiance,
            self.realtime_config.render.gamma,
            self.realtime_config.render.normalization,
            self.realtime_config.render.normalization_value,
        )
        filtered = self._filter_gpu_radiance(radiance, direct)
        display = _tone_map(
            filtered,
            self.realtime_config.render.gamma,
            self.realtime_config.render.normalization,
            self.realtime_config.render.normalization_value,
        )
        self.current_artifact = RenderArtifact(
            radiance=filtered,
            display=display,
            summary="GPU progressive path tracing",
            scene=self.realtime_state["scene"],
            config_path=self.current_path,
        )

    def _filter_gpu_radiance(self, radiance: np.ndarray, direct: Optional[np.ndarray]) -> np.ndarray:
        denoise = self.realtime_config.denoise
        if not denoise.enabled or denoise.filter_name == "none":
            return radiance
        depth, normals, object_ids = _build_gbuffer(
            self.realtime_state["scene"],
            self.realtime_state["arrays"],
            int(self.realtime_state["width"]),
            int(self.realtime_state["height"]),
        )
        settings = FilterSettings(
            name=denoise.filter_name,
            radius=denoise.radius,
            sigma_spatial=denoise.sigma_spatial,
            sigma_color=denoise.sigma_color,
            sigma_depth=denoise.sigma_depth,
            sigma_normal=denoise.sigma_normal,
            strength=denoise.strength,
            preserve_object_flux=denoise.preserve_object_flux,
        )
        guide = FilterGuide(depth=depth, normals=normals, object_ids=object_ids)
        if direct is None:
            return ImageFilterService().apply(radiance, settings, guide)
        secondary = np.clip(radiance - direct, 0.0, None)
        return split_bilateral_denoise(direct, secondary, settings, guide)

    def _update_gpu_summary(self, prefix: str, fps: float, extra: str = "") -> None:
        fps_line = "GUI frame FPS: {0:.1f}".format(fps) if np.isfinite(fps) else "GUI frame FPS: final batch"
        self.summary.setPlainText(
            prefix
            + "\nBackend: Taichi Metal Monte-Carlo path tracing\n"
            + "Треугольников: {0}\n".format(len(self.realtime_state["scene"].triangles))
            + "Area источников: {0}\n".format(len(self.realtime_state["scene"].lights))
            + "Point источников: {0}\n".format(len(self.realtime_state["scene"].point_lights))
            + "Разрешение: {0}x{1}\n".format(self.realtime_state["width"], self.realtime_state["height"])
            + "Фильтр: {0}, radius={1}, strength={2:.2f}\n".format(
                self.realtime_config.denoise.filter_name if self.realtime_config.denoise.enabled else "none",
                self.realtime_config.denoise.radius,
                self.realtime_config.denoise.strength,
            )
            + fps_line
            + "\nAccumulated SPP: {0}".format(self.realtime_frame_count)
            + extra
        )

    def _reset_gpu_accumulation(self, clear_config: bool) -> None:
        if clear_config:
            self.realtime_config = None
            self.realtime_config_text = ""
            self.realtime_state = None
        self.realtime_accumulation = None
        self.realtime_direct_accumulation = None
        self.current_raw_radiance = None
        self.current_direct_radiance = None
        self.current_raw_display = None
        self.realtime_frame_count = 0

    def _rotate_camera(self, degrees: float) -> None:
        try:
            was_running = self.realtime_timer.isActive()
            if was_running:
                self.realtime_timer.stop()
            data = json.loads(self.config_editor.toPlainText())
            position = data["camera"]["position"]
            target = data["camera"]["target"]
            px, py, pz = [float(value) for value in position]
            tx, ty, tz = [float(value) for value in target]
            dx = px - tx
            dz = pz - tz
            angle = math.radians(degrees)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rotated_x = dx * cos_a - dz * sin_a
            rotated_z = dx * sin_a + dz * cos_a
            data["camera"]["position"] = [tx + rotated_x, py, tz + rotated_z]
            self.config_editor.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
            self.realtime_config_text = self.config_editor.toPlainText().strip()
            self.realtime_config = self._apply_filter_controls(
                load_config_from_text(self.realtime_config_text, base_dir=self.current_path.parent)
            )
            self.realtime_state = self.gpu_path_tracer.create_state(self.realtime_config, config_path=self.current_path)
            self._reset_gpu_accumulation(clear_config=False)
            if was_running:
                self.realtime_button.setText("GPU PathTrace стоп")
                self._render_realtime_frame()
                self.realtime_timer.start()
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка камеры", str(error))

    def _sync_filter_controls_from_json(self) -> None:
        try:
            config = load_config_from_text(self.config_editor.toPlainText(), base_dir=self.current_path.parent)
        except Exception:
            return
        self.denoise_enabled.setChecked(config.denoise.enabled)
        if self.filter_combo.findText(config.denoise.filter_name) >= 0:
            self.filter_combo.setCurrentText(config.denoise.filter_name)
        self.filter_radius.setValue(int(config.denoise.radius))
        self.filter_strength.setValue(max(float(config.denoise.strength), 0.95 if config.denoise.filter_name == "bilateral" else 0.0))

    def _apply_filter_controls(self, config):
        denoise = DenoiseSettings(
            enabled=self.denoise_enabled.isChecked() and self.filter_combo.currentText() != "none",
            filter_name=self.filter_combo.currentText(),
            radius=int(self.filter_radius.value()),
            sigma_spatial=config.denoise.sigma_spatial,
            sigma_color=config.denoise.sigma_color,
            sigma_depth=config.denoise.sigma_depth,
            sigma_normal=config.denoise.sigma_normal,
            strength=float(self.filter_strength.value()),
            preserve_object_flux=config.denoise.preserve_object_flux,
        )
        return replace(config, denoise=denoise)

    def _on_filter_controls_changed(self, *args) -> None:
        if self.current_raw_radiance is None or self.realtime_config is None or self.realtime_state is None:
            return
        self.realtime_config = self._apply_filter_controls(self.realtime_config)
        self._set_gpu_artifact(self.current_raw_radiance, self.current_direct_radiance)
        self._update_gpu_summary(
            "GPU progressive path tracing",
            fps=float("nan"),
            extra="\nФильтр применен к уже накопленному SPP={0}.".format(self.realtime_frame_count),
        )
        self._show_image(self.current_artifact.display)

    def _on_save_ppm(self) -> None:
        if self.current_artifact is None:
            QMessageBox.information(self, "Сохранение", "Сначала выполните рендер.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить PPM", "", "PPM (*.ppm)")
        if not path:
            return
        save_ppm(Path(path), self.current_artifact.display)

    def _on_save_png(self) -> None:
        if self.current_artifact is None:
            QMessageBox.information(self, "Сохранение", "Сначала выполните рендер.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить PNG", "", "PNG (*.png)")
        if not path:
            return
        save_png(Path(path), self.current_artifact.display)

    def _on_save_hdr(self) -> None:
        if self.current_artifact is None:
            QMessageBox.information(self, "Сохранение", "Сначала выполните рендер.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить HDR", "", "Radiance HDR (*.hdr)")
        if not path:
            return
        save_hdr(Path(path), self.current_artifact.radiance)

    def _show_image(self, image: np.ndarray) -> None:
        if (
            self.compare_checkbox.isChecked()
            and self.current_raw_display is not None
            and self.current_artifact is not None
            and image.shape == self.current_raw_display.shape
        ):
            image = _comparison_image(self.current_raw_display, image)
        height, width, _ = image.shape
        buffer = np.ascontiguousarray(image.astype(np.uint8))
        qimage = QImage(buffer.data, width, height, width * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        self.preview.setPixmap(
            pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0d1420;
                color: #e6eef9;
                font-size: 13px;
            }
            QTextEdit#configEditor, QTextEdit#summaryPanel {
                background: #111b28;
                border: 1px solid #284056;
                border-radius: 16px;
                padding: 10px;
            }
            QPushButton {
                background: #17324a;
                border: 1px solid #2f5c82;
                border-radius: 12px;
                padding: 10px 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #21476a;
            }
            """
        )

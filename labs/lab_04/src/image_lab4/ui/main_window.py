from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from image_lab4.io.config_loader import load_config, load_config_from_text
from image_lab4.report.exporters import save_hdr, save_png, save_ppm
from image_lab4.services.path_tracer import PathTracer


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.current_path = Path(__file__).resolve().parents[3] / "examples" / "default_scene.json"
        self.current_artifact = None
        self.tracer = PathTracer()
        self.setWindowTitle("ЛР 4 - Трассировка путей")
        self.resize(1600, 980)
        self._build_ui()
        self._load_demo()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        controls = QHBoxLayout()
        self.demo_button = QPushButton("Демо")
        self.load_button = QPushButton("Открыть JSON")
        self.preview_button = QPushButton("Быстрый превью")
        self.render_button = QPushButton("Финальный рендер")
        self.save_ppm_button = QPushButton("Сохранить PPM")
        self.save_png_button = QPushButton("Сохранить PNG")
        self.save_hdr_button = QPushButton("Сохранить HDR")
        for button in (
            self.demo_button,
            self.load_button,
            self.preview_button,
            self.render_button,
            self.save_ppm_button,
            self.save_png_button,
            self.save_hdr_button,
        ):
            controls.addWidget(button)
        left_layout.addLayout(controls)
        self.config_editor = QTextEdit()
        self.config_editor.setObjectName("configEditor")
        left_layout.addWidget(self.config_editor, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.preview = QLabel("После рендера здесь появится изображение.\nДля быстрого старта используй кнопку 'Быстрый превью'.")
        self.preview.setMinimumSize(640, 640)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("QLabel { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #101925, stop:1 #1a2a39); color: #d9e8ff; border: 1px solid #24364c; border-radius: 18px; }")
        self.preview.setScaledContents(False)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setObjectName("summaryPanel")
        right_layout.addWidget(self.preview, stretch=5)
        right_layout.addWidget(self.summary, stretch=2)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([700, 900])

        self.demo_button.clicked.connect(self._load_demo)
        self.load_button.clicked.connect(self._on_open)
        self.preview_button.clicked.connect(lambda: self._on_render(preview=True))
        self.render_button.clicked.connect(self._on_render)
        self.save_ppm_button.clicked.connect(self._on_save_ppm)
        self.save_png_button.clicked.connect(self._on_save_png)
        self.save_hdr_button.clicked.connect(self._on_save_hdr)
        self._apply_style()

    def _load_demo(self) -> None:
        self.config_editor.setPlainText(self.current_path.read_text(encoding="utf-8"))

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть конфиг", "", "JSON (*.json)")
        if not path:
            return
        self.current_path = Path(path)
        self.config_editor.setPlainText(self.current_path.read_text(encoding="utf-8"))

    def _on_render(self, preview: bool = False) -> None:
        try:
            config = load_config_from_text(self.config_editor.toPlainText(), base_dir=self.current_path.parent)
            strict_resolution = True
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
                )
                strict_resolution = False
            self.current_artifact = self.tracer.render(
                config,
                config_path=self.current_path,
                strict_resolution=strict_resolution,
            )
            self.summary.setPlainText(self.current_artifact.summary)
            self._show_image(self.current_artifact.display)
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка рендера", str(error))

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

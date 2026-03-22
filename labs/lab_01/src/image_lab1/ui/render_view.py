from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from image_lab1.services.observer_renderer import ObserverRenderResult


class RenderView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._source_pixmap = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.summary_label = QLabel(
            "После расчета здесь появится 2D-рендер треугольника из точки зрения наблюдателя."
        )
        self.summary_label.setWordWrap(True)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(420, 420)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "QLabel { background: #06090d; border: 1px solid #1f3447; border-radius: 12px; padding: 8px; }"
        )

        layout.addWidget(self.summary_label)
        layout.addWidget(self.image_label, stretch=1)

    def set_render_result(self, render_result: ObserverRenderResult) -> None:
        height, width, _ = render_result.image.shape
        bytes_per_line = 3 * width
        image = QImage(
            render_result.image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()
        self._source_pixmap = QPixmap.fromImage(image)
        self._update_pixmap()
        self.summary_label.setText(
            f"{render_result.viewport_label}. Закрашенных пикселей поверхности: {render_result.hit_pixels}."
        )

    def clear_render(self) -> None:
        self._source_pixmap = None
        self.image_label.clear()
        self.summary_label.setText(
            "После расчета здесь появится 2D-рендер треугольника из точки зрения наблюдателя."
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        self.image_label.setPixmap(
            self._source_pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

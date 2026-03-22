from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from image_lab1.ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from image_lab4.ui.main_window import MainWindow


def main() -> None:
    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    application.exec()


if __name__ == "__main__":
    main()


import os
import sys

os.environ["QT_API"] = "pyside6"

from PySide6.QtWidgets import QApplication, QMessageBox
from interface.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    app.setApplicationName("HoloSketch 3D")
    app.setStyle("Fusion")

    try:
        window = MainWindow()
    except Exception as error:
        QMessageBox.critical(
            None,
            "Error al iniciar HoloSketch 3D",
            str(error),
        )
        raise

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

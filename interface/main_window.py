
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config.settings import (
    APP_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
)

from interface.camera_widget import CameraWidget
from interface.viewport_3d import Viewport3D


# =========================================
# MODOS DE INTERACCION
# =========================================

MODES = [
    "DRAW",
    "MOVE",
    "ROTATE",
    "NAVIGATE",
]

MODE_LABELS = {
    "DRAW": "DIBUJAR",
    "MOVE": "MOVER",
    "ROTATE": "ROTAR",
    "NAVIGATE": "CAMARA",
}


# =========================================
# ESTILO MINIMALISTA
# =========================================

STYLE = """
    QMainWindow, QWidget#central {
        background-color: #000000;
    }

    QPushButton {
        color: #bbbbbb;
        background-color: #111111;
        border: 1px solid #292929;
        border-radius: 7px;
        padding: 6px 16px;
        font-size: 12px;
        font-weight: 600;
    }

    QPushButton:hover {
        background-color: #202020;
        color: #ffffff;
    }

    QPushButton:pressed {
        background-color: #303030;
    }

    QPushButton#mode {
        background-color: #ffffff;
        color: #000000;
    }
"""


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_TITLE)

        self.resize(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        )

        self.setMinimumSize(800, 500)

        self._setup_ui()
        self._setup_shortcuts()
        self._refresh_mode()

    # =========================================
    # CREAR BOTONES
    # =========================================

    def _button(self, text, tooltip):

        button = QPushButton(text)

        button.setToolTip(tooltip)

        button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        button.setFocusPolicy(
            Qt.FocusPolicy.NoFocus
        )

        button.setMinimumHeight(34)

        return button

    # =========================================
    # INTERFAZ
    # =========================================

    def _setup_ui(self):

        central = QWidget()
        central.setObjectName("central")

        self.setCentralWidget(central)

        self.setStyleSheet(STYLE)

        root = QVBoxLayout(central)

        root.setContentsMargins(
            10, 10, 10, 10
        )

        root.setSpacing(8)

        # =====================================
        # CAMARA + VISOR 3D
        # =====================================

        content = QHBoxLayout()
        content.setSpacing(10)

        # Camara izquierda
        self.camera_widget = CameraWidget()

        self.camera_widget.setFixedWidth(340)

        # Visor 3D derecha
        self.viewport = Viewport3D()

        content.addWidget(
            self.camera_widget,
            0,
            Qt.AlignmentFlag.AlignTop,
        )

        content.addWidget(
            self.viewport,
            1,
        )

        root.addLayout(content, 1)

        # =====================================
        # BARRA INFERIOR
        # =====================================

        bar = QHBoxLayout()
        bar.setSpacing(8)

        self.mode_button = self._button(
            "",
            "Espacio - Cambiar modo de interaccion",
        )

        self.mode_button.setObjectName("mode")

        undo_button = self._button(
            "DESHACER",
            "Ctrl+Z - Eliminar el ultimo trazo",
        )

        clear_button = self._button(
            "LIMPIAR",
            "Borrar todos los trazos",
        )

        reset_button = self._button(
            "CENTRAR",
            "R - Restablecer la vista",
        )

        buttons = (
            self.mode_button,
            undo_button,
            clear_button,
            reset_button,
        )

        for button in buttons:
            bar.addWidget(button)

        bar.addStretch()

        root.addLayout(bar)

        # =====================================
        # CONEXIONES
        # =====================================

        # MediaPipe -> Motor 3D
        self.camera_widget.hand_data.connect(
            self.viewport.handle_hand_data
        )

        # Cambiar modo
        self.mode_button.clicked.connect(
            self.toggle_mode
        )

        # Deshacer
        undo_button.clicked.connect(
            self.viewport.undo
        )

        # Limpiar
        clear_button.clicked.connect(
            self.viewport.clear
        )

        # Centrar camara
        reset_button.clicked.connect(
            self.viewport.reset_view
        )

    # =========================================
    # ATAJOS DE TECLADO
    # =========================================

    def _setup_shortcuts(self):

        self.shortcuts = []

        actions = [
            (
                "D",
                lambda: self.change_mode("DRAW"),
            ),
            (
                "M",
                lambda: self.change_mode("MOVE"),
            ),
            (
                "T",
                lambda: self.change_mode("ROTATE"),
            ),
            (
                "N",
                lambda: self.change_mode("NAVIGATE"),
            ),
            (
                "Space",
                self.toggle_mode,
            ),
            (
                "R",
                self.viewport.reset_view,
            ),
            (
                "Ctrl+Z",
                self.viewport.undo,
            ),
        ]

        for key, callback in actions:

            shortcut = QShortcut(
                QKeySequence(key),
                self,
            )

            shortcut.setContext(
                Qt.ShortcutContext.ApplicationShortcut
            )

            shortcut.activated.connect(
                callback
            )

            self.shortcuts.append(
                shortcut
            )

    # =========================================
    # CAMBIAR MODO
    # =========================================

    def change_mode(self, mode):

        if mode not in MODES:
            return

        self.viewport.set_mode(mode)

        self._refresh_mode()

    # =========================================
    # ALTERNAR ENTRE LOS 4 MODOS
    # =========================================

    def toggle_mode(self):

        current_mode = self.viewport.mode

        if current_mode not in MODES:
            self.change_mode("DRAW")
            return

        index = MODES.index(
            current_mode
        )

        next_index = (
            index + 1
        ) % len(MODES)

        next_mode = MODES[
            next_index
        ]

        self.change_mode(
            next_mode
        )

    # =========================================
    # ACTUALIZAR TEXTO DEL BOTON
    # =========================================

    def _refresh_mode(self):

        current_mode = self.viewport.mode

        label = MODE_LABELS.get(
            current_mode,
            "DESCONOCIDO",
        )

        self.mode_button.setText(
            f"MODO: {label}"
        )

    # =========================================
    # CERRAR APLICACION
    # =========================================

    def closeEvent(self, event):

        self.camera_widget.close_camera()

        self.viewport.close_viewport()

        super().closeEvent(event)

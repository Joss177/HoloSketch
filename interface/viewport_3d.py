import pyvista as pv

from PySide6.QtCore import QEvent, QTimer
from PySide6.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from drawing.drawing_manager import DrawingManager


RENDER_INTERVAL_MS = 33          # ~30 fps maximo
RIGHT_RELEASE_GRACE_FRAMES = 3   # evita soltar el objeto por un parpadeo
LABEL_MS = 1500

# Zoom con la rueda del mouse
ZOOM_STEP = 0.9      # factor por "click" de rueda
ZOOM_MIN = 0.8       # parallel_scale minimo (maximo acercamiento)
ZOOM_MAX = 8.0       # parallel_scale maximo (maximo alejamiento)

CURSOR_IDLE = pv.Color("#60a5fa").float_rgb
CURSOR_GRAB = pv.Color("#fde047").float_rgb


class Viewport3D(QWidget):

    # Mano derecha:
    #   pinch  -> rota la figura
    #   puño   -> agarra y mueve la figura
    # NAVIGATE: la mano derecha (pinch) orbita la camara
    MODES = ("ROTATE", "MOVE", "NAVIGATE")
    ALIASES = {"DRAW": "ROTATE"}   # compatibilidad con botones antiguos

    def __init__(self, parent=None):
        super().__init__(parent)

        self.mode = "ROTATE"
        self.last_position = None
        self._dirty = True
        self._closed = False

        # None | "rotate" | "move"
        self._right_action = None
        self._right_released = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)

        # Captura la rueda del mouse para el zoom
        self.plotter.installEventFilter(self)

        self.setup_scene()

        self.drawing = DrawingManager(self.plotter)
        self.drawing.on_shape = self._flash_label

        self._setup_right_cursor()

        self._label_timer = QTimer(self)
        self._label_timer.setSingleShot(True)
        self._label_timer.timeout.connect(self._clear_label)

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._render_if_dirty)
        self.render_timer.start(RENDER_INTERVAL_MS)

    # ---------- escena ----------

    def setup_scene(self):
        self.plotter.set_background("#000000")

        plane = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=6,
            j_size=4,
            i_resolution=12,
            j_resolution=8,
        )

        self.plotter.add_mesh(
            plane,
            color="#0a0a0a",
            show_edges=True,
            edge_color="#303030",
            lighting=False,
            name="drawing_plane",
            reset_camera=False,
            render=False,
        )

        self.plotter.add_axes()
        self.reset_view()

    def reset_view(self):
        self.plotter.camera_position = [
            (0, 0, 12),
            (0, 0, 0),
            (0, 1, 0),
        ]
        self.plotter.camera.parallel_projection = True
        self.plotter.camera.parallel_scale = 2.6
        self._dirty = True

    def _setup_right_cursor(self):
        self.right_cursor = self.plotter.add_mesh(
            pv.Sphere(radius=0.065, theta_resolution=10, phi_resolution=10),
            color="#60a5fa",
            name="right_cursor",
            lighting=False,
            reset_camera=False,
            render=False,
        )
        self.right_cursor.SetVisibility(False)

    def _update_right_cursor(self, point, active):
        self.right_cursor.SetPosition(
            float(point[0]), float(point[1]), float(point[2]) + 0.05
        )
        self.right_cursor.GetProperty().SetColor(
            *(CURSOR_GRAB if active else CURSOR_IDLE)
        )
        self.right_cursor.SetVisibility(True)

    # ---------- zoom con rueda del mouse ----------

    def eventFilter(self, obj, event):
        if obj is self.plotter and event.type() == QEvent.Type.Wheel:
            self.zoom(event.angleDelta().y())
            return True          # consumido: VTK no hace su propio zoom
        return super().eventFilter(obj, event)

    def zoom(self, delta):
        if self._closed:
            return

        camera = self.plotter.camera
        scale = camera.parallel_scale * (ZOOM_STEP ** (delta / 120.0))
        camera.parallel_scale = max(ZOOM_MIN, min(ZOOM_MAX, scale))
        self._dirty = True

    # ---------- render ----------

    def _render_if_dirty(self):
        if self._closed or not self._dirty:
            return
        self._dirty = False
        self.plotter.render()

    # ---------- modos ----------

    def set_mode(self, mode):
        mode = self.ALIASES.get(mode, mode)
        if mode not in self.MODES:
            return

        # La mano izquierda siempre dibuja; el modo solo afecta a la derecha
        self.drawing.scene.release()

        self.last_position = None
        self._right_action = None
        self._right_released = 0
        self.mode = mode
        self._dirty = True

    # ---------- datos de ambas manos ----------

    def handle_hand_data(self, data):
        if self._closed:
            return

        data = data or {}
        left = data.get("left")
        right = data.get("right")

        # Mano izquierda: dibujar
        if left is not None:
            self.drawing.update(left["x"], left["y"], left["pinch"])
        else:
            if self.drawing.is_drawing:
                self.drawing.finish_stroke()
            self.drawing.hide_cursor()

        # Mano derecha: manipular / navegar
        if right is not None:
            point = self.drawing.map_to_3d(right["x"], right["y"])
            grabbing = right.get("grab", False)
            self._update_right_cursor(point, right["pinch"] or grabbing)

            if self.mode == "NAVIGATE":
                self.navigate(right["x"], right["y"], right["pinch"])
            else:
                if grabbing:
                    action = "move"       # puño -> agarrar y mover
                elif right["pinch"]:
                    action = "rotate"     # pinch -> rotar
                else:
                    action = None
                self.manipulate(point, action)
        else:
            self.right_cursor.SetVisibility(False)
            self.drawing.scene.release()
            self.last_position = None
            self._right_action = None
            self._right_released = 0

        self._dirty = True

    # ---------- mano derecha: mover / rotar ----------

    def manipulate(self, point, action):
        scene = self.drawing.scene

        if action is not None:
            self._right_released = 0

            # Inicio, o cambio de gesto (pinch <-> puño):
            # suelta y vuelve a agarrar
            if action != self._right_action:
                scene.release()
                self._right_action = action
                scene.grab(point)
                return

            if not scene.holding:
                return

            if action == "rotate":
                scene.rotate(point)
            else:
                scene.move(point)
            return

        # Sin gesto: suelta solo tras varios fotogramas seguidos
        if self._right_action is not None:
            self._right_released += 1
            if self._right_released >= RIGHT_RELEASE_GRACE_FRAMES:
                self._right_action = None
                self._right_released = 0
                scene.release()

    # ---------- mano derecha: navegar ----------

    def navigate(self, x, y, pinch):
        if not pinch:
            self.last_position = None
            return

        if self.last_position is None:
            self.last_position = (x, y)
            return

        last_x, last_y = self.last_position
        dx = max(-0.08, min(0.08, x - last_x))
        dy = max(-0.08, min(0.08, y - last_y))
        self.last_position = (x, y)

        camera = self.plotter.camera
        camera.Azimuth(-dx * 180)
        camera.Elevation(dy * 180)
        camera.OrthogonalizeViewUp()

        self._dirty = True

    # ---------- deshacer / limpiar ----------

    def undo(self):
        self.drawing.undo()
        self._dirty = True

    def clear(self):
        self.drawing.clear()
        self._dirty = True

    # ---------- etiqueta con el nombre de la figura ----------

    def _flash_label(self, name):
        if self._closed:
            return

        self.plotter.add_text(
            name,
            position="upper_left",
            font_size=12,
            color="white",
            name="shape_label",
            render=False,
        )

        self._label_timer.start(LABEL_MS)   # reinicia si habia otra
        self._dirty = True

    def _clear_label(self):
        if self._closed:
            return

        self.plotter.remove_actor("shape_label", render=False)
        self._dirty = True

    # ---------- cierre ----------

    def close_viewport(self):
        if self._closed:
            return

        self._closed = True
        self._label_timer.stop()
        self.render_timer.stop()
        self.plotter.close()
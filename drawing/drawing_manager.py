import traceback

import numpy as np
import pyvista as pv

from drawing.shape_recognizer import recognize
from drawing.shape_factory import build_mesh
from engine.shape_scene import ShapeScene


SHAPE_COLORS = [
    "#38dbbb",
    "#f59e0b",
    "#a78bfa",
    "#f472b6",
    "#60a5fa",
    "#34d399",
]

# Fotogramas seguidos sin pinch antes de cerrar el trazo.
# Evita que un parpadeo de la deteccion parta el dibujo en trozos.
RELEASE_GRACE_FRAMES = 4
MIN_STROKE_POINTS = 4


class DrawingManager:
    """
    Motor de dibujo HoloSketch 3D (mano izquierda).

    - Cursor y vista previa
    - Reconocimiento de figuras -> objetos 3D
    - Fallback a trazo libre
    - Deshacer y limpiar
    """

    def __init__(self, plotter):
        self.plotter = plotter

        self.strokes = []
        self.is_drawing = False
        self.counter = 0

        self.color = "#ffffff"
        self.min_distance_sq = 0.012 ** 2
        self.max_points = 1500
        self.stroke_radius = 0.028

        self.shape_mode = True
        self.on_shape = None

        self._released_frames = 0

        # Buffer de puntos (la ultima casilla guarda la punta de vista previa)
        self._buf = np.zeros((self.max_points + 1, 3), dtype=np.float32)
        self._idx = np.arange(self.max_points + 1, dtype=np.int64)
        self._n = 0
        self._last = None

        self.scene = ShapeScene(self.plotter)

        # Cursor
        self.cursor_position = None
        self.cursor_actor = self.plotter.add_mesh(
            pv.Sphere(radius=0.065, theta_resolution=10, phi_resolution=10),
            color="#ffffff",
            name="hand_cursor",
            lighting=False,
            reset_camera=False,
            render=False,
        )
        self.cursor_actor.SetVisibility(False)

        # Vista previa
        self._preview = pv.PolyData(np.zeros((2, 3), dtype=np.float32))
        self._preview.lines = np.array([2, 0, 1], dtype=np.int64)
        self._preview_actor = self.plotter.add_mesh(
            self._preview,
            name="active_stroke",
            color=self.color,
            line_width=4,
            render_lines_as_tubes=True,
            lighting=False,
            reset_camera=False,
            render=False,
        )
        self._preview_actor.SetVisibility(False)

    # ---------- coordenadas ----------

    @staticmethod
    def map_to_3d(x, y):
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))
        return np.array(
            [(x - 0.5) * 6.0, (0.5 - y) * 4.0, 0.20], dtype=np.float32
        )

    # ---------- cursor ----------

    def update_cursor(self, x, y):
        point = self.map_to_3d(x, y)
        self.cursor_position = point
        self.cursor_actor.SetPosition(*(float(v) for v in point))
        self.cursor_actor.SetVisibility(True)
        return point

    def hide_cursor(self):
        self.cursor_actor.SetVisibility(False)
        self.cursor_position = None

    # ---------- dibujo ----------

    def update(self, x, y, pinch):
        point = self.update_cursor(x, y)

        if not pinch:
            if self.is_drawing:
                self._released_frames += 1
                if self._released_frames >= RELEASE_GRACE_FRAMES:
                    self.finish_stroke()
            return

        self._released_frames = 0

        if not self.is_drawing:
            self._begin(point)
            return

        delta = point - self._last
        if float(delta @ delta) >= self.min_distance_sq:
            if self._n >= self.max_points:
                self.finish_stroke()
                self._begin(point)
                return

            self._buf[self._n] = point
            self._n += 1
            self._last = point.copy()

        self._update_preview(point)

    def _begin(self, point):
        self.is_drawing = True
        self._released_frames = 0

        self._buf[0] = point
        self._n = 1
        self._last = point.copy()

        self._preview_actor.SetVisibility(True)
        self._update_preview(point)

    def _update_preview(self, tip):
        n = self._n
        if n >= len(self._buf):
            return

        self._buf[n] = tip
        total = n + 1

        lines = np.empty(total + 1, dtype=np.int64)
        lines[0] = total
        lines[1:] = self._idx[:total]

        self._preview.points = self._buf[:total].copy()
        self._preview.lines = lines
        self._preview.Modified()

    # ---------- finalizar ----------

    def finish_stroke(self):
        if not self.is_drawing:
            return

        self.is_drawing = False
        self._released_frames = 0

        n = self._n
        pts = self._buf[:n].copy()
        self._n = 0
        self._last = None

        try:
            print(f"[DRAW] Finalizando trazo | Puntos: {n}")

            if n < MIN_STROKE_POINTS:
                print("[DRAW] Trazo demasiado corto")
                return

            created = False

            if self.shape_mode:
                try:
                    shape = recognize(pts)
                    if shape is not None:
                        created = self._add_shape(shape)
                except Exception as error:
                    print("[SHAPE] Error de reconocimiento:", error)
                    traceback.print_exc()

            if not created:
                self._add_freehand(pts)

            print(
                "[SCENE] Objetos 3D:", len(self.scene.objects),
                "| Trazos:", len(self.strokes),
            )
        finally:
            self._preview_actor.SetVisibility(False)

    # ---------- figura 3D ----------

    def _add_shape(self, shape):
        name = f"shape_{self.counter}"
        color = SHAPE_COLORS[self.counter % len(SHAPE_COLORS)]

        try:
            mesh, center = build_mesh(shape)

            if mesh is None or mesh.n_points == 0 or mesh.n_cells == 0:
                raise ValueError("Malla vacia")

            if not np.isfinite(np.asarray(mesh.points)).all():
                raise ValueError("Malla con coordenadas invalidas")

            obj = self.scene.add(
                name=name, mesh=mesh, center_xy=center, color=color
            )
            if obj is None or obj.actor is None:
                raise RuntimeError("ShapeScene.add no creo el objeto")

            self.strokes.append(
                {"name": name, "type": "shape", "color": color}
            )
            self.counter += 1

            print(
                f"[SHAPE] 3D creado: {name} ({shape.name}) | "
                f"Vertices: {mesh.n_points}"
            )

            if self.on_shape:
                try:
                    self.on_shape(shape.name)
                except Exception as error:
                    print("[UI] Error mostrando nombre:", error)

            return True

        except Exception as error:
            print(f"[SHAPE] Fallo creando {name}: {error}")
            traceback.print_exc()
            self.scene.remove(name)
            return False

    # ---------- trazo libre ----------

    def _add_freehand(self, pts):
        if len(pts) < 2:
            return False

        name = f"stroke_{self.counter}"

        try:
            lines = np.empty(len(pts) + 1, dtype=np.int64)
            lines[0] = len(pts)
            lines[1:] = np.arange(len(pts), dtype=np.int64)

            poly = pv.PolyData(np.asarray(pts, dtype=np.float32))
            poly.lines = lines

            tube = poly.tube(radius=self.stroke_radius, n_sides=6)
            if tube.n_points == 0:
                raise ValueError("Tubo sin vertices")

            actor = self.plotter.add_mesh(
                tube,
                name=name,
                color=self.color,
                lighting=False,
                reset_camera=False,
                render=False,
            )
            actor.SetVisibility(True)

            self.strokes.append(
                {"name": name, "type": "freehand", "color": self.color}
            )
            self.counter += 1
            print(f"[DRAW] Trazo libre guardado: {name}")
            return True

        except Exception as error:
            print("[DRAW] Error guardando trazo:", error)
            traceback.print_exc()
            return False

    # ---------- deshacer / limpiar ----------

    def undo(self):
        if self.is_drawing:
            self.finish_stroke()

        if not self.strokes:
            return

        last = self.strokes.pop()
        name = last["name"]

        if last["type"] == "shape":
            self.scene.remove(name)
        else:
            self.plotter.remove_actor(name, render=False)

        print(f"[UNDO] Eliminado: {name}")

    def clear(self):
        self.is_drawing = False
        self._released_frames = 0
        self._n = 0
        self._last = None

        self._preview_actor.SetVisibility(False)

        for stroke in self.strokes:
            if stroke["type"] == "freehand":
                self.plotter.remove_actor(stroke["name"], render=False)

        self.scene.clear()
        self.strokes.clear()

        print("[DRAW] Escena limpiada")
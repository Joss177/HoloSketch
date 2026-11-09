import numpy as np
import pyvista as pv

HIGHLIGHT = pv.Color("#fde047").float_rgb
ROT_SENSITIVITY = 60.0     # grados por unidad de escena (6 unidades ~ 360 grados)
MAX_STEP = 0.35            # limite de salto por fotograma (unidades de escena)
MOVE_SMOOTH = 0.6          # 0..1, mayor = mas directo, menor = mas suave
PICK_MARGIN = 0.35         # margen extra al seleccionar con la mano
BOUNDS_X = 3.0
BOUNDS_Y = 2.0


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _orthonormalize(r):
    u, _, vt = np.linalg.svd(r)
    return u @ vt


class SceneShape:
    __slots__ = ("name", "actor", "pos", "rot", "radius", "rgb")


class ShapeScene:

    def __init__(self, plotter):
        self.plotter = plotter
        self.objects = []
        self.held = None
        self._last = None
        self._offset = None

    @property
    def holding(self):
        return self.held is not None

    # ---------- transformacion ----------

    @staticmethod
    def _apply(obj):
        m = np.eye(4)
        m[:3, :3] = obj.rot
        m[:3, 3] = obj.pos
        obj.actor.user_matrix = m

    # ---------- crear ----------

    def add(self, name, mesh, center_xy, color, z=0.2, tilt=(-22, 28)):
        if mesh is None or mesh.n_points == 0:
            return None

        if any(o.name == name for o in self.objects):
            self.remove(name)

        # Centrar la malla en el origen para rotar sobre su centro
        center = np.asarray(mesh.center)
        if np.linalg.norm(center) > 1e-5:
            mesh = mesh.copy()
            mesh.translate(-center, inplace=True)

        actor = self.plotter.add_mesh(
            mesh,
            color=color,
            name=name,
            specular=0.3,
            reset_camera=False,
            render=False,
        )

        obj = SceneShape()
        obj.name = name
        obj.actor = actor
        obj.pos = np.array([center_xy[0], center_xy[1], z], dtype=np.float64)
        obj.rot = _rot_y(tilt[1]) @ _rot_x(tilt[0])   # inclinacion inicial

        b = mesh.bounds
        obj.radius = max(
            0.25, 0.5 * float(np.hypot(b[1] - b[0], b[3] - b[2]))
        )
        obj.rgb = pv.Color(color).float_rgb

        self._apply(obj)
        actor.SetVisibility(True)

        self.objects.append(obj)
        print(f"[SHAPE] Creado: {name}")
        return obj

    # ---------- seleccion ----------

    def _pick(self, point):
        x, y = float(point[0]), float(point[1])
        best, best_d = None, float("inf")

        for obj in self.objects:
            d = float(np.hypot(obj.pos[0] - x, obj.pos[1] - y))
            if d <= obj.radius + PICK_MARGIN and d < best_d:
                best, best_d = obj, d

        return best

    def grab(self, point):
        obj = self._pick(point)

        print(
            "[RIGHT] Seleccion en", np.round(point[:2], 2),
            "| Objetos:", len(self.objects),
            "| Encontrado:", obj.name if obj else None,
        )

        if obj is None:
            return False

        self.release()

        self.held = obj
        self._last = np.asarray(point, dtype=np.float64).copy()
        self._offset = obj.pos[:2] - np.asarray(point[:2], dtype=np.float64)

        obj.actor.GetProperty().SetColor(*HIGHLIGHT)
        return True

    def release(self):
        if self.held is not None:
            self.held.actor.GetProperty().SetColor(*self.held.rgb)

        self.held = None
        self._last = None
        self._offset = None

    # ---------- mover ----------

    def move(self, point):
        if self.held is None:
            return

        obj = self.held
        target = np.asarray(point[:2], dtype=np.float64) + self._offset
        target[0] = np.clip(target[0], -BOUNDS_X, BOUNDS_X)
        target[1] = np.clip(target[1], -BOUNDS_Y, BOUNDS_Y)

        delta = (target - obj.pos[:2]) * MOVE_SMOOTH
        norm = float(np.hypot(*delta))
        if norm > MAX_STEP:
            delta *= MAX_STEP / norm

        obj.pos[:2] += delta
        self._apply(obj)

    # ---------- rotar ----------

    def rotate(self, point):
        if self.held is None:
            return

        point = np.asarray(point, dtype=np.float64)

        if self._last is None:
            self._last = point.copy()
            return

        dx = float(np.clip(point[0] - self._last[0], -MAX_STEP, MAX_STEP))
        dy = float(np.clip(point[1] - self._last[1], -MAX_STEP, MAX_STEP))
        self._last = point.copy()

        # Sin zona muerta por fotograma: se acumula el movimiento lento
        if dx == 0.0 and dy == 0.0:
            return

        obj = self.held
        delta = _rot_x(-dy * ROT_SENSITIVITY) @ _rot_y(dx * ROT_SENSITIVITY)
        obj.rot = _orthonormalize(delta @ obj.rot)
        self._apply(obj)

    # ---------- eliminar ----------

    def remove(self, name):
        for i, obj in enumerate(self.objects):
            if obj.name == name:
                if self.held is obj:
                    self.release()

                self.plotter.remove_actor(name, render=False)
                self.objects.pop(i)
                return True

        return False

    def clear(self):
        self.release()

        for obj in self.objects:
            self.plotter.remove_actor(obj.name, render=False)

        self.objects.clear()
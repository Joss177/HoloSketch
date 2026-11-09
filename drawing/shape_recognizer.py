import math
from dataclasses import dataclass

import cv2
import numpy as np

MIN_PERIMETER = 0.8        # trazos mas pequenos se ignoran (unidades de escena)
MIN_RAW_POINTS = 6         # puntos minimos del trazo original
RESAMPLE_N = 64            # puntos tras remuestrear el trazo
CLOSE_RATIO = 0.22         # separacion inicio-fin maxima para considerarlo cerrado
LINE_STRAIGHTNESS = 0.06   # desviacion maxima de una linea recta
MAX_FIT_ERROR = 0.30       # error maximo (relativo) para aceptar una figura
RIGHT_ANGLE_TOL = 14.0     # grados de tolerancia para angulos rectos
PARALLEL_TOL = 12.0        # grados de tolerancia para lados paralelos
AXIS_SNAP_DEG = 10.0       # se endereza si esta a menos de esto de los ejes

POLY_NAMES = {3: "TRIANGULO", 5: "PENTAGONO", 6: "HEXAGONO"}


@dataclass
class Shape2D:
    name: str
    points: np.ndarray     # (n, 2) vertices limpios
    kind: str              # "poly" | "line"


# ---------- utilidades ----------

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _rotate(pts, center, ang):
    c, s = math.cos(ang), math.sin(ang)
    r = np.array([[c, -s], [s, c]])
    return (pts - center) @ r.T + center


def _snap_axes(pts, direction):
    """Si la figura esta casi recta, la endereza exactamente."""
    ang = math.degrees(math.atan2(direction[1], direction[0]))
    diff = round(ang / 90) * 90 - ang
    if abs(diff) <= AXIS_SNAP_DEG:
        return _rotate(pts, pts.mean(axis=0), math.radians(diff))
    return pts


def _ccw(poly):
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return poly if area > 0 else poly[::-1].copy()


def _regular(center, radius, n, start):
    a = start + np.arange(n) * 2 * math.pi / n
    return center + radius * np.column_stack((np.cos(a), np.sin(a)))


def _resample(pts, n=RESAMPLE_N):
    """Remuestrea el trazo a n puntos equidistantes (quita el sesgo
    de velocidad de la mano y reduce el ruido)."""
    pts = pts.astype(np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = np.concatenate(([0.0], np.cumsum(seg)))
    total = dist[-1]
    if total < 1e-9:
        return pts, 0.0
    t = np.linspace(0.0, total, n)
    out = np.column_stack((
        np.interp(t, dist, pts[:, 0]),
        np.interp(t, dist, pts[:, 1]),
    ))
    return out, float(total)


def _poly_error(pts, poly):
    cnt = poly.astype(np.float32).reshape(-1, 1, 2)
    return float(np.mean([
        abs(cv2.pointPolygonTest(cnt, (float(x), float(y)), True))
        for x, y in pts
    ]))


def _ellipse_error(pts, ell):
    (cx, cy), (w, h), ang = ell
    a, b = max(w / 2, 1e-6), max(h / 2, 1e-6)
    t = math.radians(ang)
    d = pts - np.array([cx, cy])
    xr = d[:, 0] * math.cos(t) + d[:, 1] * math.sin(t)
    yr = -d[:, 0] * math.sin(t) + d[:, 1] * math.cos(t)
    return float(np.mean(np.abs(np.sqrt((xr / a) ** 2 + (yr / b) ** 2) - 1.0)))


# ---------- API ----------

def recognize(points3d):
    raw = np.asarray(points3d, dtype=np.float64)

    if raw.ndim != 2 or raw.shape[1] < 2:
        print("[SHAPE] Coordenadas invalidas")
        return None

    raw = raw[:, :2]

    if len(raw) < MIN_RAW_POINTS:
        print(f"[SHAPE] Muy pocos puntos ({len(raw)})")
        return None

    pts, perimeter = _resample(raw)

    print(f"[SHAPE] Puntos: {len(raw)} | Perimetro: {perimeter:.2f}")

    if perimeter < MIN_PERIMETER:
        print("[SHAPE] Trazo demasiado pequeno")
        return None

    gap = float(np.linalg.norm(pts[0] - pts[-1]))
    close_limit = CLOSE_RATIO * perimeter

    print(f"[SHAPE] Cierre: {gap:.3f} | Limite: {close_limit:.3f}")

    result = _line(pts) if gap > close_limit else _closed(pts)

    if result is None:
        print("[SHAPE] No reconocida. Se conserva el trazo libre.")
    else:
        print(f"[SHAPE] Reconocida: {result.name}")

    return result


def _line(pts):
    d = pts[-1] - pts[0]
    length = float(np.linalg.norm(d))
    if length < 0.4:
        return None
    n = np.array([-d[1], d[0]]) / length
    deviation = float(np.abs((pts - pts[0]) @ n).max())
    if deviation / length > LINE_STRAIGHTNESS:
        return None                       # curva abierta: queda a mano alzada
    seg = np.array([pts[0], pts[-1]], dtype=np.float64)
    return Shape2D("LINEA", _snap_axes(seg, d), "line")


def _closed(pts):
    cv_pts = pts.astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(cv_pts)
    area = float(cv2.contourArea(hull))
    if area < 1e-3:
        return None

    approx = cv2.approxPolyDP(
        hull, 0.04 * cv2.arcLength(hull, True), True
    ).reshape(-1, 2).astype(np.float64)
    k = len(approx)
    mean_r = math.sqrt(area / math.pi)

    poly_err = _poly_error(pts, approx) / mean_r if k >= 3 else 9.0

    try:
        ell = cv2.fitEllipse(cv_pts)
        ell_err = _ellipse_error(pts, ell)
    except cv2.error:
        ell, ell_err = None, 9.0

    if min(poly_err, ell_err) > MAX_FIT_ERROR:
        return None                       # demasiado feo: no adivinar

    is_ellipse = ell is not None and (
        k >= 7 or k < 3 or ell_err < (0.7 if k <= 4 else 1.3) * poly_err
    )

    if is_ellipse:
        return _ellipse(ell)
    if k < 3 or k > 6:
        return _ellipse(ell) if ell is not None else None
    return _polygon(approx, k)


def _ellipse(ell):
    (cx, cy), (w, h), ang = ell
    a, b = w / 2, h / 2
    if abs(a - b) / max(a, b) < 0.12:
        a = b = (a + b) / 2
        name = "CIRCULO"
    else:
        name = "ELIPSE"
        snapped = round(ang / 90) * 90
        if abs(ang - snapped) <= AXIS_SNAP_DEG:
            ang = snapped

    t = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    th = math.radians(ang)
    ux = np.array([math.cos(th), math.sin(th)])
    uy = np.array([-math.sin(th), math.cos(th)])
    poly = (np.array([cx, cy])
            + np.outer(a * np.cos(t), ux) + np.outer(b * np.sin(t), uy))
    return Shape2D(name, poly, "poly")


def _polygon(approx, k):
    poly = _ccw(approx)
    if k == 4:
        return _quad(poly)

    edges = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    if edges.std() / edges.mean() < 0.12:           # lados casi iguales -> regular
        c = poly.mean(axis=0)
        r = float(np.linalg.norm(poly - c, axis=1).mean())
        a0 = math.degrees(math.atan2(poly[0][1] - c[1], poly[0][0] - c[0]))
        step = 360 / k
        snapped = 90 + round((a0 - 90) / step) * step   # un vertice hacia arriba
        if abs(a0 - snapped) <= AXIS_SNAP_DEG:
            a0 = snapped
        poly = _regular(c, r, k, math.radians(a0))
    return Shape2D(POLY_NAMES[k], poly, "poly")


def _quad(poly):
    e = np.roll(poly, -1, axis=0) - poly
    lens = np.linalg.norm(e, axis=1)
    u = e / lens[:, None]

    def parallel(i, j):
        return math.degrees(
            math.acos(min(1.0, abs(float(u[i] @ u[j]))))
        ) < PARALLEL_TOL

    p02, p13 = parallel(0, 2), parallel(1, 3)
    sin_tol = math.sin(math.radians(RIGHT_ANGLE_TOL))
    right = all(abs(float(u[i] @ u[(i + 1) % 4])) < sin_tol for i in range(4))

    # --- rectangulo / cuadrado ---
    if p02 and p13 and right:
        (cx, cy), (w, h), ang = cv2.minAreaRect(poly.astype(np.float32))
        if abs(w - h) / max(w, h) < 0.15:
            w = h = (w + h) / 2
            name = "CUADRADO"
        else:
            name = "RECTANGULO"
        snapped = round(ang / 90) * 90
        if abs(ang - snapped) <= AXIS_SNAP_DEG:
            ang = snapped
        t = math.radians(ang)
        ux = np.array([math.cos(t), math.sin(t)]) * w / 2
        uy = np.array([-math.sin(t), math.cos(t)]) * h / 2
        c = np.array([cx, cy])
        return Shape2D(
            name,
            np.array([c - ux - uy, c + ux - uy, c + ux + uy, c - ux + uy]),
            "poly",
        )

    # --- paralelogramo / rombo ---
    if p02 and p13:
        a, b = (e[0] - e[2]) / 2, (e[1] - e[3]) / 2
        v = np.array([[0, 0], a, a + b, b])
        v = v - v.mean(axis=0) + poly.mean(axis=0)
        name = "ROMBO" if lens.std() / lens.mean() < 0.15 else "PARALELOGRAMO"
        return Shape2D(name, _snap_axes(v, a), "poly")

    # --- trapecio ---
    if p02 or p13:
        i = 0 if p02 else 1
        d = _unit(u[i] - u[(i + 2) % 4])
        la, lc = lens[i], lens[(i + 2) % 4]
        ma = (poly[i] + poly[(i + 1) % 4]) / 2
        mc = (poly[(i + 2) % 4] + poly[(i + 3) % 4]) / 2
        new = poly.copy()
        new[i] = ma - d * la / 2
        new[(i + 1) % 4] = ma + d * la / 2
        new[(i + 2) % 4] = mc + d * lc / 2
        new[(i + 3) % 4] = mc - d * lc / 2
        return Shape2D("TRAPECIO", _snap_axes(new, d), "poly")

    return Shape2D("CUADRILATERO", poly, "poly")
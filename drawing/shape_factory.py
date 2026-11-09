import numpy as np
import pyvista as pv

SPHERE_RES = 20
TUBE_SIDES = 6
MIN_SIZE = 0.05

PRISMS = {
    "TRAPECIO",
    "PENTAGONO",
    "HEXAGONO",
    "ROMBO",
    "PARALELOGRAMO",
    "CUADRILATERO",
}


def _angle_deg(local):
    return float(np.degrees(np.arctan2(
        local[1, 1] - local[0, 1],
        local[1, 0] - local[0, 0],
    )))


def build_mesh(shape, depth=None):
    """
    Convierte Shape2D en un solido 3D.
    Devuelve (mesh_local, center_xy).
    """
    pts = np.asarray(shape.points, dtype=np.float64)

    if (
        pts.ndim != 2
        or pts.shape[1] != 2
        or len(pts) < 2
        or not np.isfinite(pts).all()
    ):
        raise ValueError("Puntos de figura invalidos")

    center = pts.mean(axis=0)
    local = pts - center
    name = shape.name.upper()

    width, height = np.ptp(local, axis=0)
    width = max(float(width), MIN_SIZE)
    height = max(float(height), MIN_SIZE)

    # LINEA -> TUBO
    if shape.kind == "line" or name == "LINEA":
        mesh = pv.Line(
            (float(local[0, 0]), float(local[0, 1]), 0.0),
            (float(local[-1, 0]), float(local[-1, 1]), 0.0),
        ).tube(radius=0.035, n_sides=TUBE_SIDES)

    # CUADRADO -> CUBO
    elif name == "CUADRADO":
        size = max(float(np.linalg.norm(local[1] - local[0])), MIN_SIZE)
        mesh = pv.Cube(x_length=size, y_length=size, z_length=size)
        mesh.rotate_z(_angle_deg(local), inplace=True)

    # RECTANGULO -> PRISMA RECTANGULAR
    elif name == "RECTANGULO":
        w = max(float(np.linalg.norm(local[1] - local[0])), MIN_SIZE)
        h = max(float(np.linalg.norm(local[2] - local[1])), MIN_SIZE)
        thickness = float(depth) if depth is not None else min(w, h) * 0.6
        mesh = pv.Cube(
            x_length=w, y_length=h, z_length=max(thickness, MIN_SIZE)
        )
        mesh.rotate_z(_angle_deg(local), inplace=True)

    # CIRCULO -> ESFERA
    elif name == "CIRCULO":
        radius = max(float(np.linalg.norm(local, axis=1).mean()), MIN_SIZE)
        mesh = pv.Sphere(
            radius=radius,
            theta_resolution=SPHERE_RES,
            phi_resolution=SPHERE_RES,
        )

    # ELIPSE -> ELIPSOIDE (PCA para ejes y orientacion)
    elif name in ("ELIPSE", "OVALO", "\u00d3VALO"):
        covariance = np.cov(local.T)
        values, vectors = np.linalg.eigh(covariance)

        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], 1e-6)
        vectors = vectors[:, order]

        a, b = np.sqrt(2.0 * values)

        mesh = pv.Sphere(
            radius=1.0,
            theta_resolution=SPHERE_RES,
            phi_resolution=SPHERE_RES,
        )

        new_pts = np.array(mesh.points, dtype=np.float64)
        xy = new_pts[:, :2] * np.array([a, b])
        new_pts[:, :2] = xy @ vectors.T
        new_pts[:, 2] *= max(min(a, b), MIN_SIZE)
        mesh.points = new_pts

    # TRIANGULO -> PIRAMIDE (con base)
    elif name == "TRIANGULO":
        base = local[:3]
        side = float(np.linalg.norm(base[1] - base[0]))
        pyr_h = max(
            float(depth) if depth is not None else float(np.clip(side, 0.3, 1.5)),
            MIN_SIZE,
        )

        bottom = np.column_stack((base, np.full(3, -pyr_h / 2)))
        apex = np.array([[0.0, 0.0, pyr_h / 2]])
        vertices = np.vstack((bottom, apex))

        faces = np.array([
            3, 0, 2, 1,   # base
            3, 0, 1, 3,
            3, 1, 2, 3,
            3, 2, 0, 3,
        ])
        mesh = pv.PolyData(vertices, faces=faces)

    # OTROS POLIGONOS -> PRISMAS
    elif name in PRISMS:
        if len(local) < 3:
            raise ValueError("Poligono sin suficientes vertices")

        thickness = (
            float(depth)
            if depth is not None
            else float(np.clip(min(width, height) * 0.55, 0.2, 1.0))
        )

        area2 = np.sum(
            local[:, 0] * np.roll(local[:, 1], -1)
            - np.roll(local[:, 0], -1) * local[:, 1]
        )
        poly_pts = local if area2 >= 0 else local[::-1].copy()
        n = len(poly_pts)

        vertices = np.column_stack((poly_pts, np.full(n, -thickness / 2)))
        faces = np.hstack(([n], np.arange(n)))

        polygon = pv.PolyData(vertices, faces=faces)
        mesh = polygon.extrude((0, 0, thickness), capping=True)

    else:
        raise ValueError(f"Figura no soportada: {name}")

    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"Malla vacia para {name}")

    return mesh, center
"""Geometria dell'array UMA-16 e griglie di focalizzazione."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Posizioni UMA-16 (ordine canali USB) come nel file acoular 'minidsp_uma-16.xml'
# (griglia 4x4, passo 42 mm, centrata nell'origine, z = 0). Usate se acoular non e' disponibile.
UMA16_POS = np.array([
    [0.021, -0.063], [0.063, -0.063], [0.021, -0.021], [0.063, -0.021],
    [0.021, 0.021], [0.063, 0.021], [0.021, 0.063], [0.063, 0.063],
    [-0.063, 0.063], [-0.021, 0.063], [-0.063, 0.021], [-0.021, 0.021],
    [-0.063, -0.021], [-0.021, -0.021], [-0.063, -0.063], [-0.021, -0.063],
]).T  # (2,16)

INTERNAL = 'interna (UMA-16)'


def acoular_xml_dir() -> Path | None:
    try:
        import acoular as ac
        return Path(ac.__file__).parent / 'xml'
    except Exception:
        return None


def available_geometries() -> list[str]:
    out = [INTERNAL]
    d = acoular_xml_dir()
    if d is not None and d.exists():
        uma = sorted(p.name for p in d.glob('*.xml') if 'uma' in p.name.lower())
        out = uma + out
    return out


def _read_xml(path: Path) -> np.ndarray:
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    pts = [[float(e.get('x')), float(e.get('y')), float(e.get('z'))] for e in root.iter('pos')]
    return np.array(pts).T


def load_positions(name: str, flip_x=False, flip_y=False, swap_xy=False) -> np.ndarray:
    """Restituisce posizioni microfoni (3, 16) in metri, con trasformazioni di orientamento."""
    if name == INTERNAL:
        pos = np.vstack([UMA16_POS, np.zeros((1, 16))])
    else:
        p = Path(name)
        if not p.is_absolute():
            d = acoular_xml_dir()
            p = (d / name) if d is not None else p
        pos = _read_xml(p) if p.exists() else np.vstack([UMA16_POS, np.zeros((1, 16))])
    return transform_positions(pos, flip_x, flip_y, swap_xy)


def transform_positions(pos, flip_x=False, flip_y=False, swap_xy=False):
    pos = np.array(pos, dtype=float).copy()
    if swap_xy:
        pos[[0, 1]] = pos[[1, 0]]
    if flip_x:
        pos[0] *= -1
    if flip_y:
        pos[1] *= -1
    return pos


ORIENTATIONS = [(fx, fy, sw) for sw in (False, True) for fx in (False, True) for fy in (False, True)]


class FocusGrid:
    """Griglia di punti di focalizzazione.

    - 'Emisfero (u,v)': coseni direttori u = sin(th)cos(phi), v = sin(th)sin(phi); l'asse z e' la
      normale dell'array (verso il cielo se l'array guarda in alto). Adatto a droni (campo lontano).
    - 'Piano (x,y @ z)': piano parallelo all'array a distanza z (come bf_example_app).
    """

    def __init__(self, mode, res, x_min, x_max, y_min, y_max, z, max_off_axis_deg=80.0):
        self.mode = mode
        self.z = float(z)
        if mode.startswith('Emisfero'):
            lim = 1.0
            self.x = np.arange(-lim, lim + 1e-9, res)
            self.y = np.arange(-lim, lim + 1e-9, res)
        else:
            self.x = np.arange(x_min, x_max + 1e-9, res)
            self.y = np.arange(y_min, y_max + 1e-9, res)
        X, Y = np.meshgrid(self.x, self.y, indexing='xy')   # (ny, nx)
        self.X, self.Y = X, Y
        if mode.startswith('Emisfero'):
            r2 = X ** 2 + Y ** 2
            smax = np.sin(np.radians(max_off_axis_deg))
            self.valid = r2 <= smax ** 2
            W = np.sqrt(np.clip(1 - r2, 0, None))
            dirs = np.stack([X, Y, W], -1)
            self.points = dirs * self.z                      # punti a distanza nominale z
        else:
            self.valid = np.ones_like(X, bool)
            self.points = np.stack([X, Y, np.full_like(X, self.z)], -1)
        self.shape = X.shape
        self.flat_points = self.points.reshape(-1, 3)
        self.flat_valid = self.valid.ravel()

    @property
    def extent(self):
        return float(self.x[0]), float(self.x[-1]), float(self.y[0]), float(self.y[-1])

    @property
    def step(self):
        return float(self.x[1] - self.x[0]) if len(self.x) > 1 else 1.0

    def to_angles(self, gx, gy):
        """Coordinate griglia -> (azimut, elevazione sopra il piano dell'array, angolo dall'asse) in gradi."""
        if self.mode.startswith('Emisfero'):
            r = min(np.hypot(gx, gy), 1.0)
            w = np.sqrt(1 - r ** 2)
            vx, vy, vz = gx, gy, w
        else:
            vx, vy, vz = gx, gy, self.z
        az = np.degrees(np.arctan2(vy, vx))
        el = np.degrees(np.arctan2(vz, np.hypot(vx, vy)))
        return az, el, 90.0 - el

    def from_position(self, p):
        """Posizione 3D (m) di una sorgente -> coordinate di griglia (gx, gy)."""
        p = np.asarray(p, float)
        if self.mode.startswith('Emisfero'):
            d = p / np.linalg.norm(p)
            return d[0], d[1]
        # proiezione della direzione sul piano z
        return p[0] * self.z / p[2], p[1] * self.z / p[2]

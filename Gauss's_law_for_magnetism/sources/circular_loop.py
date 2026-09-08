from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LoopParameters:
    """
    Parámetros físicos y geométricos de una espira circular.

    Attributes
    ----------
    radius : float
        Radio de la espira [m].
    current : float
        Corriente eléctrica [A].
    num_segments : int
        Número de segmentos para discretizar la espira.
    center : np.ndarray | None
        Centro de la espira en coordenadas cartesianas [m].
    normal : np.ndarray | None
        Vector normal a la espira. Por defecto +z.
    """
    radius: float
    current: float
    num_segments: int
    center: np.ndarray | None = None
    normal: np.ndarray | None = None


class CircularLoop:
    """
    Espira circular en 3D.

    Convención:
    - La espira está inicialmente en el plano xy.
    - La corriente positiva recorre la espira en sentido antihorario
      vista desde +z.
    """

    def __init__(
        self,
        radius: float,
        current: float,
        num_segments: int,
        center: np.ndarray | None = None,
        normal: np.ndarray | None = None,
    ) -> None:
        if radius <= 0:
            raise ValueError("El radio debe ser positivo.")
        if num_segments < 3:
            raise ValueError("num_segments debe ser al menos 3.")

        self.radius = float(radius)
        self.current = float(current)
        self.num_segments = int(num_segments)

        self.center = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)
        if self.center.shape != (3,):
            raise ValueError("center debe tener forma (3,)")

        self.normal = np.array([0.0, 0.0, 1.0], dtype=float) if normal is None else np.asarray(normal, dtype=float)
        if self.normal.shape != (3,):
            raise ValueError("normal debe tener forma (3,)")

    def angles(self) -> np.ndarray:
        """
        Ángulos de discretización en [0, 2pi).
        """
        return np.linspace(0.0, 2.0 * np.pi, self.num_segments, endpoint=False)

    def points(self) -> np.ndarray:
        """
        Devuelve los puntos discretos de la espira.

        Returns
        -------
        ndarray, shape (N, 3)
            Puntos de la espira.
        """
        phi = self.angles()
        x = self.radius * np.cos(phi)
        y = self.radius * np.sin(phi)
        z = np.zeros_like(phi)

        pts = np.column_stack((x, y, z))
        return pts + self.center

    def tangent_vectors(self) -> np.ndarray:
        """
        Vectores tangentes analíticos d r / d phi.
        """
        phi = self.angles()
        dx_dphi = -self.radius * np.sin(phi)
        dy_dphi = self.radius * np.cos(phi)
        dz_dphi = np.zeros_like(phi)

        tangents = np.column_stack((dx_dphi, dy_dphi, dz_dphi))
        return tangents

    def segment_vectors(self) -> np.ndarray:
        """
        Vectores segmento de la poligonal cerrada.
        """
        pts = self.points()
        next_pts = np.roll(pts, shift=-1, axis=0)
        return next_pts - pts

    def midpoints(self) -> np.ndarray:
        """
        Puntos medios de cada segmento.
        """
        pts = self.points()
        next_pts = np.roll(pts, shift=-1, axis=0)
        return 0.5 * (pts + next_pts)

    def radial_deviation(self) -> np.ndarray:
        """
        Desviación radial |sqrt(x^2 + y^2) - R| para cada punto.
        """
        pts = self.points() - self.center
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        return np.abs(radii - self.radius)

    def closure_error(self) -> float:
        """
        Norma del error de cierre de la poligonal.
        """
        segs = self.segment_vectors()
        return float(np.linalg.norm(np.sum(segs, axis=0)))

    def normal_unit(self) -> np.ndarray:
        """
        Devuelve el vector normal unitario de la espira.
        """
        n = np.asarray(self.normal, dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0:
            raise ValueError("El vector normal no puede ser nulo.")
        return n / norm

    def summary(self) -> dict:
        """
        Resumen geométrico útil para validación.
        """
        dev = self.radial_deviation()
        return {
            "radius": self.radius,
            "current": self.current,
            "num_segments": self.num_segments,
            "center": self.center.copy(),
            "normal": self.normal_unit(),
            "max_radial_deviation": float(np.max(dev)),
            "mean_radial_deviation": float(np.mean(dev)),
            "closure_error": self.closure_error(),
        }
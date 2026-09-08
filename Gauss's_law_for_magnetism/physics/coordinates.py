from __future__ import annotations

import numpy as np


def cartesian_to_cylindrical(points: np.ndarray) -> np.ndarray:
    """
    Convierte coordenadas cartesianas a cilíndricas.

    Parameters
    ----------
    points : ndarray, shape (..., 3)
        Coordenadas cartesianas [x, y, z].

    Returns
    -------
    ndarray, shape (..., 3)
        Coordenadas cilíndricas [rho, phi, z],
        donde phi está en el intervalo [-pi, pi].
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError("Se esperan puntos con tres componentes: x, y, z.")

    x = pts[..., 0]
    y = pts[..., 1]
    z = pts[..., 2]

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return np.stack((rho, phi, z), axis=-1)


def cylindrical_to_cartesian(points: np.ndarray) -> np.ndarray:
    """
    Convierte coordenadas cilíndricas a cartesianas.

    Parameters
    ----------
    points : ndarray, shape (..., 3)
        Coordenadas cilíndricas [rho, phi, z].

    Returns
    -------
    ndarray, shape (..., 3)
        Coordenadas cartesianas [x, y, z].
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError("Se esperan puntos con tres componentes: rho, phi, z.")

    rho = pts[..., 0]
    phi = pts[..., 1]
    z = pts[..., 2]

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return np.stack((x, y, z), axis=-1)


def spherical_to_cartesian(points: np.ndarray) -> np.ndarray:
    """
    Convierte coordenadas esféricas a cartesianas.

    Convención:
    - r: radio
    - theta: ángulo polar medido desde +z
    - phi: ángulo azimutal en el plano xy

    Parameters
    ----------
    points : ndarray, shape (..., 3)
        Coordenadas esféricas [r, theta, phi].

    Returns
    -------
    ndarray, shape (..., 3)
        Coordenadas cartesianas [x, y, z].
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError("Se esperan puntos con tres componentes: r, theta, phi.")

    r = pts[..., 0]
    theta = pts[..., 1]
    phi = pts[..., 2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack((x, y, z), axis=-1)


def cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    """
    Convierte coordenadas cartesianas a esféricas.

    Returns
    -------
    ndarray, shape (..., 3)
        Coordenadas esféricas [r, theta, phi].
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError("Se esperan puntos con tres componentes: x, y, z.")

    x = pts[..., 0]
    y = pts[..., 1]
    z = pts[..., 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / np.where(r == 0, 1.0, r), -1.0, 1.0))
    phi = np.arctan2(y, x)

    return np.stack((r, theta, phi), axis=-1)


def cylindrical_unit_vectors(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve los vectores unitarios cilíndricos e_rho, e_phi, e_z
    expresados en base cartesiana.

    Parameters
    ----------
    phi : ndarray
        Ángulo azimutal.

    Returns
    -------
    (e_rho, e_phi, e_z) : tuple of ndarray
        Cada uno con forma (..., 3).
    """
    phi = np.asarray(phi, dtype=float)

    e_rho = np.stack((np.cos(phi), np.sin(phi), np.zeros_like(phi)), axis=-1)
    e_phi = np.stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=-1)
    e_z = np.stack((np.zeros_like(phi), np.zeros_like(phi), np.ones_like(phi)), axis=-1)

    return e_rho, e_phi, e_z
from __future__ import annotations

import numpy as np
from scipy.constants import mu_0

from .errors import relative_error


def analytical_bz_on_axis(radius: float, current: float, z: np.ndarray) -> np.ndarray:
    """
    Campo magnético analítico sobre el eje z de una espira circular.

    Parameters
    ----------
    radius : float
        Radio de la espira [m].
    current : float
        Corriente [A].
    z : ndarray
        Coordenada axial.

    Returns
    -------
    ndarray
        Bz analítico sobre el eje.
    """
    z = np.asarray(z, dtype=float)
    return mu_0 * current * radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)


def compare_on_axis(
    radius: float,
    current: float,
    z: np.ndarray,
    numerical_bz: np.ndarray,
) -> dict:
    """
    Compara el campo numérico con el analítico sobre el eje z.

    Returns
    -------
    dict
        Diccionario con campo analítico, error relativo y estadísticas simples.
    """
    analytical = analytical_bz_on_axis(radius, current, z)
    rel = relative_error(numerical_bz, analytical)

    return {
        "analytical": analytical,
        "relative_error": rel,
        "max_relative_error": float(np.max(rel)),
        "mean_relative_error": float(np.mean(rel)),
        "rms_relative_error": float(np.sqrt(np.mean(rel**2))),
    }
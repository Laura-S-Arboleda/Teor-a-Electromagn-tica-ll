from __future__ import annotations

import numpy as np


def trapezoidal_rule_1d(f: np.ndarray, dx: float) -> float:
    """
    Integra una función 1D sobre una malla uniforme usando la regla del trapecio.

    Parameters
    ----------
    f : ndarray
        Valores de la función.
    dx : float
        Paso uniforme.

    Returns
    -------
    float
        Aproximación de la integral.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 1:
        raise ValueError("f debe ser un arreglo 1D.")
    if f.size < 2:
        raise ValueError("Se requieren al menos dos puntos para integrar.")

    return float(dx * (0.5 * f[0] + np.sum(f[1:-1]) + 0.5 * f[-1]))


def cumulative_trapezoidal_1d(f: np.ndarray, dx: float) -> np.ndarray:
    """
    Integral acumulada 1D mediante la regla del trapecio.

    Parameters
    ----------
    f : ndarray
        Valores de la función.
    dx : float
        Paso uniforme.

    Returns
    -------
    ndarray
        Integral acumulada con el mismo tamaño que f.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 1:
        raise ValueError("f debe ser un arreglo 1D.")
    if f.size < 2:
        raise ValueError("Se requieren al menos dos puntos para integrar.")

    cum = np.zeros_like(f)
    cum[1:] = np.cumsum(0.5 * (f[:-1] + f[1:]) * dx)
    return cum
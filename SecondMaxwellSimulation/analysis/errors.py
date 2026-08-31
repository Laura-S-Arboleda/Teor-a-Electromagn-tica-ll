from __future__ import annotations

import numpy as np


def max_abs_error(values: np.ndarray) -> float:
    """
    Máximo valor absoluto de un campo escalar.
    """
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values)))


def mean_abs_error(values: np.ndarray) -> float:
    """
    Error medio absoluto.
    """
    values = np.asarray(values, dtype=float)
    return float(np.mean(np.abs(values)))


def rms_error(values: np.ndarray) -> float:
    """
    Error cuadrático medio (RMS).
    """
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def relative_error(numerical: np.ndarray, analytical: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Error relativo punto a punto.

    Parameters
    ----------
    numerical : ndarray
        Valores numéricos.
    analytical : ndarray
        Valores analíticos.
    eps : float
        Umbral para evitar división por cero.

    Returns
    -------
    ndarray
        Error relativo.
    """
    numerical = np.asarray(numerical, dtype=float)
    analytical = np.asarray(analytical, dtype=float)

    denom = np.maximum(np.abs(analytical), eps)
    return np.abs(numerical - analytical) / denom


def nonzero_fraction(values: np.ndarray, tol: float = 1e-10) -> float:
    """
    Fracción de puntos cuyo valor absoluto supera una tolerancia.
    """
    values = np.asarray(values, dtype=float)
    return float(np.mean(np.abs(values) > tol))
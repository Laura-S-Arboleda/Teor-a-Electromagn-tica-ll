from __future__ import annotations

import numpy as np


def error_statistics(values: np.ndarray) -> dict:
    """
    Devuelve estadísticas básicas de un campo de error.
    """
    values = np.asarray(values, dtype=float)
    abs_values = np.abs(values)

    return {
        "max_abs": float(np.max(abs_values)),
        "mean_abs": float(np.mean(abs_values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "median_abs": float(np.median(abs_values)),
        "std": float(np.std(values)),
    }


def locate_largest_errors(values: np.ndarray, threshold_fraction: float = 0.9):
    """
    Devuelve índices donde el valor absoluto supera una fracción del máximo.
    """
    values = np.asarray(values, dtype=float)
    abs_values = np.abs(values)
    max_val = np.max(abs_values)

    if max_val == 0:
        return np.empty((0, values.ndim), dtype=int)

    return np.argwhere(abs_values >= threshold_fraction * max_val)
from __future__ import annotations

import numpy as np


def error_statistics(values: np.ndarray) -> dict:
    """
    Devuelve estadísticas básicas de un campo de error,
    ignorando valores NaN e infinitos.
    """
    values = np.asarray(values, dtype=float)

    # Conservar únicamente valores finitos
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "max_abs": np.nan,
            "mean_abs": np.nan,
            "rms": np.nan,
            "median_abs": np.nan,
            "std": np.nan,
        }

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
    Devuelve índices donde el valor absoluto supera una fracción del máximo,
    ignorando valores NaN e infinitos.
    """
    values = np.asarray(values, dtype=float)
    abs_values = np.abs(values)

    # Evitar que NaN/inf afecten la búsqueda
    finite_mask = np.isfinite(values)

    if not np.any(finite_mask):
        return np.empty((0, values.ndim), dtype=int)

    max_val = np.max(abs_values[finite_mask])

    if max_val == 0:
        return np.empty((0, values.ndim), dtype=int)

    mask = finite_mask & (
        abs_values >= threshold_fraction * max_val
    )

    return np.argwhere(mask)
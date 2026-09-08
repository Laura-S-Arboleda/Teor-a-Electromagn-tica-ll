from __future__ import annotations

import numpy as np


def magnetic_field_magnitude(B: np.ndarray) -> np.ndarray:
    """
    Calcula la magnitud |B| de un campo vectorial magnético.

    Parameters
    ----------
    B : ndarray, shape (..., 3)
        Campo magnético.

    Returns
    -------
    ndarray
        Magnitud del campo en cada punto.
    """
    B = np.asarray(B, dtype=float)
    if B.shape[-1] != 3:
        raise ValueError("Se espera un campo vectorial con 3 componentes.")
    return np.linalg.norm(B, axis=-1)


def normalize_vector_field(B: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Normaliza un campo vectorial evitando división por cero.

    Parameters
    ----------
    B : ndarray, shape (..., 3)
        Campo vectorial.
    eps : float
        Umbral para evitar división por cero.

    Returns
    -------
    ndarray
        Campo normalizado.
    """
    B = np.asarray(B, dtype=float)
    if B.shape[-1] != 3:
        raise ValueError("Se espera un campo vectorial con 3 componentes.")

    norm = np.linalg.norm(B, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return B / norm
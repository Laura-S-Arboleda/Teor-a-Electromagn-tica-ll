from __future__ import annotations

import numpy as np


def divergence_from_grid(
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:

    Bx = np.asarray(Bx, dtype=float)
    By = np.asarray(By, dtype=float)
    Bz = np.asarray(Bz, dtype=float)

    if Bx.shape != By.shape or Bx.shape != Bz.shape:
        raise ValueError("Bx, By y Bz deben tener la misma forma.")

    divB = np.full_like(Bx, np.nan)

    divB[1:-1, 1:-1, 1:-1] = (
        (Bx[2:, 1:-1, 1:-1] - Bx[:-2, 1:-1, 1:-1])
        / (2.0 * dx)
        +
        (By[1:-1, 2:, 1:-1] - By[1:-1, :-2, 1:-1])
        / (2.0 * dy)
        +
        (Bz[1:-1, 1:-1, 2:] - Bz[1:-1, 1:-1, :-2])
        / (2.0 * dz)
    )

    return divB
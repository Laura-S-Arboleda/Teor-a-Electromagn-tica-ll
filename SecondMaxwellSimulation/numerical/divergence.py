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
    """
    Calcula la divergencia numérica de un campo vectorial en una malla cartesiana regular.

    Parameters
    ----------
    Bx, By, Bz : ndarray, shape (Nx, Ny, Nz)
        Componentes del campo magnético.
    dx, dy, dz : float
        Pasos espaciales.

    Returns
    -------
    divB : ndarray, shape (Nx, Ny, Nz)
        Divergencia numérica.
    """
    Bx = np.asarray(Bx, dtype=float)
    By = np.asarray(By, dtype=float)
    Bz = np.asarray(Bz, dtype=float)

    if Bx.shape != By.shape or Bx.shape != Bz.shape:
        raise ValueError("Bx, By y Bz deben tener la misma forma.")

    divB = np.zeros_like(Bx)

    divB[1:-1, :, :] += (Bx[2:, :, :] - Bx[:-2, :, :]) / (2.0 * dx)
    divB[:, 1:-1, :] += (By[:, 2:, :] - By[:, :-2, :]) / (2.0 * dy)
    divB[:, :, 1:-1] += (Bz[:, :, 2:] - Bz[:, :, :-2]) / (2.0 * dz)

    return divB
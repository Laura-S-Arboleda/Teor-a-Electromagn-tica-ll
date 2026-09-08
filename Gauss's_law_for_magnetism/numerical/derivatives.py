from __future__ import annotations

import numpy as np


def central_difference_1d(f: np.ndarray, dx: float) -> np.ndarray:
    """
    Derivada centrada 1D para un arreglo 1D.

    Parameters
    ----------
    f : ndarray
        Valores de la función en una malla uniforme.
    dx : float
        Paso espacial.

    Returns
    -------
    ndarray
        Derivada numérica. Los bordes se dejan en cero.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 1:
        raise ValueError("f debe ser un arreglo 1D.")

    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx)
    return df


def forward_difference_1d(f: np.ndarray, dx: float) -> np.ndarray:
    """
    Derivada hacia adelante 1D.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 1:
        raise ValueError("f debe ser un arreglo 1D.")

    df = np.zeros_like(f)
    df[:-1] = (f[1:] - f[:-1]) / dx
    df[-1] = df[-2]
    return df


def backward_difference_1d(f: np.ndarray, dx: float) -> np.ndarray:
    """
    Derivada hacia atrás 1D.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 1:
        raise ValueError("f debe ser un arreglo 1D.")

    df = np.zeros_like(f)
    df[1:] = (f[1:] - f[:-1]) / dx
    df[0] = df[1]
    return df


def gradient_3d(F: np.ndarray, dx: float, dy: float, dz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gradiente numérico de un campo escalar F definido en una malla 3D regular.

    Parameters
    ----------
    F : ndarray, shape (Nx, Ny, Nz)
        Campo escalar.
    dx, dy, dz : float
        Pasos espaciales.

    Returns
    -------
    (dFdx, dFdy, dFdz) : tuple of ndarray
        Derivadas parciales numéricas.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 3:
        raise ValueError("F debe ser un arreglo 3D.")

    dFdx = np.zeros_like(F)
    dFdy = np.zeros_like(F)
    dFdz = np.zeros_like(F)

    dFdx[1:-1, :, :] = (F[2:, :, :] - F[:-2, :, :]) / (2.0 * dx)
    dFdy[:, 1:-1, :] = (F[:, 2:, :] - F[:, :-2, :]) / (2.0 * dy)
    dFdz[:, :, 1:-1] = (F[:, :, 2:] - F[:, :, :-2]) / (2.0 * dz)

    return dFdx, dFdy, dFdz
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_scalar_field_2d(ax, X, Z, values, cmap: str = "viridis", alpha: float = 0.85):
    """
    Dibuja un campo escalar en 2D como mapa de calor.
    """
    return ax.contourf(X, Z, values, levels=50, cmap=cmap, alpha=alpha)


def plot_error_field_2d(ax, X, Z, values, cmap: str = "magma", alpha: float = 0.9):
    """
    Dibuja el valor absoluto de un campo de error en 2D.
    """
    return ax.contourf(X, Z, np.abs(values), levels=50, cmap=cmap, alpha=alpha)
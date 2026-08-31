from __future__ import annotations

import matplotlib.pyplot as plt


def plot_streamlines_2d(ax, X, Z, BX, BZ, color: str = "white", density: float = 1.2, linewidth: float = 1.0):
    """
    Dibuja líneas de flujo en 2D.
    """
    ax.streamplot(X, Z, BX, BZ, color=color, density=density, linewidth=linewidth)
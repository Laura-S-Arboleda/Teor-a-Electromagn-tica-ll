from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_vector_field_3d(ax, points: np.ndarray, vectors: np.ndarray, color: str = "navy", scale: float = 1.0):
    """
    Dibuja un campo vectorial 3D con quiver.
    """
    pts = np.asarray(points, dtype=float)
    vec = np.asarray(vectors, dtype=float)

    ax.quiver(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        vec[:, 0],
        vec[:, 1],
        vec[:, 2],
        length=scale,
        normalize=True,
        color=color,
        alpha=0.8,
    )


def plot_vector_field_2d(ax, X, Z, BX, BZ, color: str = "navy", pivot: str = "mid"):
    """
    Dibuja un campo vectorial 2D en el plano xz.
    """
    ax.quiver(X, Z, BX, BZ, color=color, pivot=pivot, angles="xy")
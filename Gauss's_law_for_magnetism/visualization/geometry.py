from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_loop_3d(ax, loop_points: np.ndarray, color: str = "crimson", linewidth: float = 2.0, label: str = "Espira"):
    """
    Dibuja la espira circular en un eje 3D.
    """
    pts = np.asarray(loop_points, dtype=float)
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        color=color,
        linewidth=linewidth,
        label=label,
    )


def plot_coordinate_axes_3d(ax, length: float = 1.5):
    """
    Dibuja ejes coordenados en 3D.
    """
    ax.quiver(0, 0, 0, length, 0, 0, color="black", arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, 0, length, 0, color="black", arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, 0, 0, length, color="black", arrow_length_ratio=0.08)

    ax.text(length, 0, 0, "x")
    ax.text(0, length, 0, "y")
    ax.text(0, 0, length, "z")


def plot_loop_2d(ax, loop_points: np.ndarray, color: str = "crimson", linewidth: float = 2.0, label: str = "Espira"):
    """
    Dibuja la proyección 2D de la espira.
    """
    pts = np.asarray(loop_points, dtype=float)
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth, label=label)
    ax.set_aspect("equal", adjustable="box")
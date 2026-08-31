from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import LoopConfig
from sources.circular_loop import CircularLoop
from fields.biot_savart import biot_savart_field
from fields.magnetic_field import magnetic_field_magnitude
from analysis.validation import compare_on_axis
from analysis.diagnostics import error_statistics
from numerical.divergence import divergence_from_grid
from visualization.geometry import plot_loop_3d, plot_coordinate_axes_3d
from visualization.vector_fields import plot_vector_field_3d
from visualization.animation import animate_loop_current, animate_loop_current_3d

_LAST_ANIMATION = None


def compute_field_on_axis(loop: CircularLoop, z_values: np.ndarray) -> np.ndarray:
    z_values = np.asarray(z_values, dtype=float)
    observation_points = np.column_stack(
        (np.zeros_like(z_values), np.zeros_like(z_values), z_values)
    )
    return biot_savart_field(
        observation_points=observation_points,
        source_points=loop.midpoints(),
        segment_vectors=loop.segment_vectors(),
        current=loop.current,
    )


def compute_field_on_grid_3d(loop: CircularLoop, extent: float = 1.5, n: int = 15):
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    z = np.linspace(-extent, extent, n)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    obs = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    B = biot_savart_field(
        observation_points=obs,
        source_points=loop.midpoints(),
        segment_vectors=loop.segment_vectors(),
        current=loop.current,
    )

    Bx = B[:, 0].reshape(X.shape)
    By = B[:, 1].reshape(X.shape)
    Bz = B[:, 2].reshape(X.shape)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    return X, Y, Z, Bx, By, Bz, dx, dy, dz


def compute_field_on_grid_xz(loop: CircularLoop, extent: float = 2.0, n: int = 35):
    x = np.linspace(-extent, extent, n)
    z = np.linspace(-extent, extent, n)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)

    obs = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    B = biot_savart_field(
        observation_points=obs,
        source_points=loop.midpoints(),
        segment_vectors=loop.segment_vectors(),
        current=loop.current,
    )

    BX = B[:, 0].reshape(X.shape)
    BZ = B[:, 2].reshape(X.shape)
    Bmag = magnetic_field_magnitude(B).reshape(X.shape)

    return X, Z, BX, BZ, Bmag


def run_on_axis_validation():
    cfg = LoopConfig(radius=1.0, current=1.0, num_segments=200)
    loop = CircularLoop(cfg.radius, cfg.current, cfg.num_segments)

    z_values = np.linspace(-3.0, 3.0, 101)
    B = compute_field_on_axis(loop, z_values)
    Bz_num = B[:, 2]

    report = compare_on_axis(loop.radius, loop.current, z_values, Bz_num)

    print("=== Validación sobre el eje z ===")
    print(f"Error relativo máximo: {report['max_relative_error']:.6e}")
    print(f"Error relativo medio:  {report['mean_relative_error']:.6e}")
    print(f"Error relativo RMS:    {report['rms_relative_error']:.6e}")

    return z_values, Bz_num, report["analytical"], report["relative_error"]


def run_divergence_analysis():
    cfg = LoopConfig(radius=1.0, current=1.0, num_segments=300)
    loop = CircularLoop(cfg.radius, cfg.current, cfg.num_segments)

    X, Y, Z, Bx, By, Bz, dx, dy, dz = compute_field_on_grid_3d(loop, extent=1.5, n=17)
    divB = divergence_from_grid(Bx, By, Bz, dx, dy, dz)
    stats = error_statistics(divB)

    print("=== Análisis numérico de divergencia ===")
    print(f"Máximo |div B|: {stats['max_abs']:.6e}")
    print(f"Mean |div B|:   {stats['mean_abs']:.6e}")
    print(f"RMS(div B):     {stats['rms']:.6e}")

    return X, Y, Z, divB


def convergence_study(
        segment_list=(20, 40, 80, 160),
        grid_list=(11, 15, 21, 31),
        extent: float = 1.5,
):
    results = []
    for nseg in segment_list:
        loop = CircularLoop(radius=1.0, current=1.0, num_segments=nseg)
        for ng in grid_list:
            X, Y, Z, Bx, By, Bz, dx, dy, dz = compute_field_on_grid_3d(loop, extent=extent, n=ng)
            divB = divergence_from_grid(Bx, By, Bz, dx, dy, dz)
            stats = error_statistics(divB)
            results.append(
                {
                    "num_segments": nseg,
                    "grid_n": ng,
                    "max_abs": stats["max_abs"],
                    "rms": stats["rms"],
                    "mean_abs": stats["mean_abs"],
                }
            )
            print(f"Nseg={nseg:4d}, grid={ng:3d} -> max={stats['max_abs']:.6e}, rms={stats['rms']:.6e}")
    return pd.DataFrame(results)


def run_convergence_analysis():
    return convergence_study()


def plot_on_axis_validation(z_values, Bz_num, Bz_an, rel_err):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(z_values, Bz_num, label="Bz numérico", color="navy")
    ax1.plot(z_values, Bz_an, "--", label="Bz analítico", color="crimson")
    ax1.set_xlabel("z [m]")
    ax1.set_ylabel("Bz [T]")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(z_values, rel_err, color="darkgreen", alpha=0.6, label="Error relativo")
    ax2.set_ylabel("Error relativo")
    ax2.legend(loc="lower right")

    plt.title("Validación del campo magnético sobre el eje z")
    plt.tight_layout()
    plt.show()


def run_visualization():
    cfg = LoopConfig(radius=1.0, current=1.0, num_segments=200)
    loop = CircularLoop(cfg.radius, cfg.current, cfg.num_segments)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    plot_loop_3d(ax, loop.points())
    plot_coordinate_axes_3d(ax, length=1.6)

    sample_points = loop.midpoints()
    sample_field = biot_savart_field(
        observation_points=sample_points,
        source_points=loop.midpoints(),
        segment_vectors=loop.segment_vectors(),
        current=loop.current,
    )
    plot_vector_field_3d(ax, sample_points, sample_field, scale=0.2)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Espira circular y campo magnético")
    ax.legend()
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()
    return loop


def run_animation(mode: str = "3d"):
    global _LAST_ANIMATION
    cfg = LoopConfig(radius=1.0, current=1.0, num_segments=200)
    loop = CircularLoop(cfg.radius, cfg.current, cfg.num_segments)

    mode = mode.lower().strip()

    if mode == "2d":
        X, Z, BX, BZ, _ = compute_field_on_grid_xz(loop, extent=2.0, n=55)
        _LAST_ANIMATION = animate_loop_current(
            loop_points=loop.points(),
            X=X,
            Z=Z,
            BX=BX,
            BZ=BZ,
            num_frames=240,
            interval=35,
        )
    else:
        X, Y, Z, Bx, By, Bz, _, _, _ = compute_field_on_grid_3d(loop, extent=1.2, n=7)
        sample_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        sample_field = np.column_stack((Bx.ravel(), By.ravel(), Bz.ravel()))
        _LAST_ANIMATION = animate_loop_current_3d(
            loop_points=loop.points(),
            sample_points=sample_points,
            sample_field=sample_field,
            num_frames=None,
            interval=35,
            rotate_camera=False,
        )

    plt.show()
    return _LAST_ANIMATION
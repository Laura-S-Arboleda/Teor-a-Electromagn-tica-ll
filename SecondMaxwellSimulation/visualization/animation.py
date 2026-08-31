from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def animate_loop_current(
        loop_points: np.ndarray,
        X: np.ndarray | None = None,
        Z: np.ndarray | None = None,
        BX: np.ndarray | None = None,
        BZ: np.ndarray | None = None,
        num_frames: int = 120,
        interval: int = 50,
        save_path: str | None = None,
        writer: str = "pillow",
):
    """
    Anima un marcador que recorre la espira y, opcionalmente, superpone líneas de
    campo magnético en el plano xz para visualizar su topología cerrada.
    """
    pts = np.asarray(loop_points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]

    fig, (ax_loop, ax_field) = plt.subplots(1, 2, figsize=(12, 5.8))

    ax_loop.plot(x, y, color="crimson", linewidth=2.5, label="Espira")
    marker, = ax_loop.plot([], [], "o", color="navy", markersize=8, label="Corriente")
    ax_loop.set_aspect("equal", adjustable="box")
    ax_loop.set_xlabel("x [m]")
    ax_loop.set_ylabel("y [m]")
    ax_loop.set_title("Corriente sobre la espira")
    ax_loop.grid(True, linestyle="--", alpha=0.4)
    ax_loop.legend()

    radius = np.max(np.sqrt(x**2 + y**2))
    ax_loop.set_xlim(-1.2 * radius, 1.2 * radius)
    ax_loop.set_ylim(-1.2 * radius, 1.2 * radius)

    ax_field.set_title("Líneas de campo magnético en plano xz (y=0)")
    ax_field.set_xlabel("x [m]")
    ax_field.set_ylabel("z [m]")
    ax_field.set_aspect("equal", adjustable="box")
    ax_field.grid(True, linestyle="--", alpha=0.35)

    if X is not None and Z is not None and BX is not None and BZ is not None:
        mag = np.sqrt(BX**2 + BZ**2)
        bg = ax_field.contourf(X, Z, mag, levels=40, cmap="viridis", alpha=0.85)
        fig.colorbar(bg, ax=ax_field, label="|B| [T]")
        ax_field.streamplot(
            X,
            Z,
            BX,
            BZ,
            color="white",
            density=1.15,
            linewidth=0.9,
            arrowsize=0.8,
        )
        ax_field.plot([-radius, radius], [0.0, 0.0], color="crimson", linewidth=2.0, label="Sección espira")
        ax_field.legend(loc="upper right")
    else:
        ax_field.text(
            0.5,
            0.5,
            "Sin datos de campo para streamlines",
            transform=ax_field.transAxes,
            ha="center",
            va="center",
        )

    def update(frame: int):
        idx = frame % len(pts)
        p = pts[idx]
        marker.set_data([p[0]], [p[1]])
        return (marker,)

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)

    if save_path is not None:
        fps = max(1, 1000 // interval)
        if writer == "ffmpeg":
            anim.save(save_path, writer=FFMpegWriter(fps=fps))
        else:
            anim.save(save_path, writer=PillowWriter(fps=fps))

    return anim


def animate_loop_current_3d(
        loop_points: np.ndarray,
        sample_points: np.ndarray,
        sample_field: np.ndarray,
        num_frames: int | None = None,
        interval: int = 40,
        rotate_camera: bool = False,
        save_path: str | None = None,
        writer: str = "pillow",
):
    """
    Animación 3D enriquecida:
    - Espira circular 3D.
    - Marcador de corriente con estela.
    - Campo B en quiver 3D coloreado por magnitud.
    - Rotación de cámara manual/automática.
    - Proyecciones del marcador en planos coordenados.
    Controles:
      [espacio] pausa/reanuda
      [r]       activa/desactiva auto-rotación
      [+/-]     cambia velocidad de rotación
      [t]       muestra/oculta estela
      [p]       muestra/oculta proyecciones
      [←/→]     rota cámara manualmente (azimut)
      [↑/↓]     inclina cámara manualmente (elevación)
    """
    pts = np.asarray(loop_points, dtype=float)
    p_sample = np.asarray(sample_points, dtype=float)
    b_sample = np.asarray(sample_field, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("loop_points debe tener forma (N, 3).")
    if p_sample.ndim != 2 or p_sample.shape[1] != 3:
        raise ValueError("sample_points debe tener forma (M, 3).")
    if b_sample.ndim != 2 or b_sample.shape[1] != 3:
        raise ValueError("sample_field debe tener forma (M, 3).")
    if p_sample.shape[0] != b_sample.shape[0]:
        raise ValueError("sample_points y sample_field deben tener igual cantidad de filas.")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="crimson", linewidth=2.5, label="Espira")
    marker, = ax.plot([], [], [], "o", color="navy", markersize=7, label="Corriente")

    # Campo magnético (estático) en puntos de muestreo
    b_norm = np.linalg.norm(b_sample, axis=1, keepdims=True)
    b_safe = np.where(b_norm > 1e-15, b_sample / b_norm, 0.0)
    ax.quiver(
        p_sample[:, 0],
        p_sample[:, 1],
        p_sample[:, 2],
        b_safe[:, 0],
        b_safe[:, 1],
        b_safe[:, 2],
        length=0.18,
        normalize=True,
        color="teal",
        alpha=0.75,
    )

    # Ejes
    max_extent = float(np.max(np.abs(pts))) * 1.4
    max_extent = max(max_extent, 1.2)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_zlim(-max_extent, max_extent)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Animación 3D: espira y campo magnético")
    ax.legend()

    def update(frame: int):
        idx = frame % len(pts)
        p = pts[idx]
        marker.set_data([p[0]], [p[1]])
        marker.set_3d_properties([p[2]])

        if rotate_camera:
            azim = (frame * 1.2) % 360.0
            ax.view_init(elev=24, azim=azim)

        return (marker,)

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)

    if save_path is not None:
        fps = max(1, 1000 // interval)
        if writer == "ffmpeg":
            anim.save(save_path, writer=FFMpegWriter(fps=fps))
        else:
            anim.save(save_path, writer=PillowWriter(fps=fps))

    return anim
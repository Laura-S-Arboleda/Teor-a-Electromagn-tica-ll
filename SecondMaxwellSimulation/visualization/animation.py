from __future__ import annotations
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from time import perf_counter

try:
    import pyvista as pv
except Exception:  # pragma: no cover - fallback si no está disponible
    pv = None


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
    Fallback 3D con Matplotlib.
    """
    pts = np.asarray(loop_points, dtype=float)
    p_sample = np.asarray(sample_points, dtype=float)
    b_sample = np.asarray(sample_field, dtype=float)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="crimson", linewidth=2.5, label="Espira")
    marker, = ax.plot([], [], [], "o", color="navy", markersize=7, label="Corriente")

    b_norm = np.linalg.norm(b_sample, axis=1, keepdims=True)
    b_safe = np.where(b_norm > 1e-15, b_sample / b_norm, 0.0)
    ax.quiver(
        p_sample[:, 0], p_sample[:, 1], p_sample[:, 2],
        b_safe[:, 0], b_safe[:, 1], b_safe[:, 2],
        length=0.18, normalize=True, color="teal", alpha=0.75,
    )

    max_extent = float(np.max(np.abs(pts))) * 1.4
    max_extent = max(max_extent, 1.2)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_zlim(-max_extent, max_extent)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Animación 3D (fallback Matplotlib)")
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

    frame_source = range(num_frames) if num_frames is not None else None
    anim = FuncAnimation(
        fig, update, frames=frame_source, interval=interval,
        blit=False, repeat=True, cache_frame_data=False
    )

    if save_path is not None:
        fps = max(1, 1000 // interval)
        if writer == "ffmpeg":
            anim.save(save_path, writer=FFMpegWriter(fps=fps))
        else:
            anim.save(save_path, writer=PillowWriter(fps=fps))

    return anim

def plot_field_lines_only_2d(X, Z, BX, BZ, radius: float = 1.0):
    fig, ax = plt.subplots(figsize=(7, 6))
    mag = np.sqrt(BX**2 + BZ**2)
    bg = ax.contourf(X, Z, mag, levels=40, cmap="viridis", alpha=0.85)
    fig.colorbar(bg, ax=ax, label="|B| [T]")

    ax.streamplot(
        X, Z, BX, BZ,
        color="white",
        density=1.15,
        linewidth=0.9,
        arrowsize=0.8,
    )
    ax.plot([-radius, radius], [0.0, 0.0], color="crimson", linewidth=2.0)
    ax.set_title("Líneas de campo magnético en plano xz (y=0)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()



def animate_loop_current_3d_pyvista(
        loop_points: np.ndarray,
        sample_points: np.ndarray,
        sample_field: np.ndarray,
        interval_ms: int = 35,
        point_size: float = 16.0,
        vector_scale: float = 0.18,
        line_wobble_amplitude: float = 0.08,
        line_wobble_period_frames: int = 90,
):
    """
    Animación 3D interactiva con PyVista.
    Controles:
      espacio: pausa/reanuda
      r: auto-rotar cámara
      + / -: velocidad de auto-rotación
      t: mostrar/ocultar estela
      q: cerrar ventana
    """
    if pv is None:
        raise ImportError("PyVista no está disponible. Instala: pyvista pyvistaqt vtk")

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

    plotter = pv.Plotter(window_size=(1280, 800))
    plotter.set_background("#f4f7fb")

    # Espira
    closed = np.vstack([pts, pts[0]])
    loop_spline = pv.Spline(closed, n_points=max(200, 4 * len(closed)))
    plotter.add_mesh(loop_spline, color="crimson", line_width=5, name="loop")

    # Campo B en glyphs (flechas)
    pdata = pv.PolyData(p_sample)
    pdata["B"] = b_sample
    bmag = np.linalg.norm(b_sample, axis=1)
    pdata["Bmag"] = bmag
    b_unit = np.where(bmag[:, None] > 1e-14, b_sample / bmag[:, None], 0.0)
    pdata["B_unit"] = b_unit
    glyphs = pdata.glyph(orient="B_unit", scale=False, factor=vector_scale)
    glyphs_actor = plotter.add_mesh(
        glyphs,
        scalars="Bmag",
        cmap="plasma",
        opacity=0.35,
        lighting=False,
        name="B_glyphs",
        show_scalar_bar=False,
    )

    # Marcador corriente
    arrow_data = pv.PolyData(pts[[0]])
    v0 = pts[1] - pts[0]
    v0 = v0 / (np.linalg.norm(v0) + 1e-15)
    arrow_data["dir"] = np.array([v0])

    arrow_glyph = arrow_data.glyph(orient="dir", scale=False, factor=0.6, geom=pv.Arrow())
    arrow_actor = plotter.add_mesh(
        arrow_glyph,
        color="gold",
        name="current_arrow",
    )

    # Estela
    trail_data = None

    # Ejes y texto
    plotter.show_grid(color="black")
    plotter.add_axes(interactive=True, line_width=2)
    plotter.add_text(
        "Controles: [space] pausa  [r] auto-rotar  [+/-] velocidad  [t] estela  [q] salir",
        position="lower_left",
        font_size=10,
        color="black",
        name="help_text",
    )

    state = {
        "idx": 0,
        "paused": False,
        "auto_rotate": False,
        "az_speed": 1.2,
        "show_trail": True,
        "trail_len": min(120, len(pts)),
        "line_motion": True,
        "frame": 0,
        "show_lines": True,
    }
    trail_actor = None

    def _save_screenshot():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{ts}.png"
        plotter.screenshot(
            filename=filename,
            transparent_background=False,
            scale=3
        )
        print(f"[INFO] Captura HD guardada: {filename}")

    def _toggle_pause():
        state["paused"] = not state["paused"]

    def _toggle_lines():
        state["show_lines"] = not state["show_lines"]
        glyphs_actor.SetVisibility(state["show_lines"])
        plotter.render()

    def _toggle_rotate():
        state["auto_rotate"] = not state["auto_rotate"]

    def _faster():
        state["az_speed"] = min(10.0, state["az_speed"] + 0.3)

    def _slower():
        state["az_speed"] = max(0.0, state["az_speed"] - 0.3)

    def _toggle_trail():
        state["show_trail"] = not state["show_trail"]

    def _toggle_line_motion():
        state["line_motion"] = not state["line_motion"]

    plotter.add_key_event("space", _toggle_pause)
    plotter.add_key_event("r", _toggle_rotate)
    plotter.add_key_event("+", _faster)
    plotter.add_key_event("-", _slower)
    plotter.add_key_event("t", _toggle_trail)
    plotter.add_key_event("s", _save_screenshot)
    plotter.add_key_event("l", _toggle_lines)
    plotter.add_key_event("m", _toggle_line_motion)

    def _tick():
        nonlocal trail_data, trail_actor
        if state["paused"]:
            return

        state["idx"] = (state["idx"] + 1) % len(pts)
        i = state["idx"]
        j = (i + 1) % len(pts)

        p = pts[i]
        t = pts[j] - pts[i]
        t = t / (np.linalg.norm(t) + 1e-15)

        arrow_data.points = np.array([p])
        arrow_data["dir"] = np.array([t])

        new_arrow = arrow_data.glyph(orient="dir", scale=False, factor=0.22, geom=pv.Arrow())
        arrow_glyph.copy_from(new_arrow)
        arrow_actor.mapper.dataset = arrow_glyph
        plotter.render()

        if state["show_trail"]:
            i0 = max(0, state["idx"] - state["trail_len"])
            seg = pts[i0:state["idx"] + 1]
            # polyline para estela
            if len(seg) >= 2:
                poly = pv.lines_from_points(seg)
                if trail_data is None:
                    trail_data = poly
                    trail_actor = plotter.add_mesh(
                        trail_data,
                        color="orange",
                        line_width=3,
                        lighting=False,
                        name="trail",
                    )
                else:
                    trail_data.copy_from(poly)
                    if trail_actor is not None:
                        trail_actor.mapper.dataset = trail_data
            elif trail_data is not None:
                trail_data.points = np.empty((0, 3))
                if trail_actor is not None:
                    trail_actor.mapper.dataset = trail_data
        elif trail_data is not None:
            trail_data.points = np.empty((0, 3))
            if trail_actor is not None:
                trail_actor.mapper.dataset = trail_data

        if state["line_motion"] and line_wobble_period_frames > 0:
            phase = 2.0 * np.pi * (state["frame"] / line_wobble_period_frames)
            wobble = 1.0 + line_wobble_amplitude * np.sin(phase)
            pdata["B_unit"] = b_unit * wobble
            glyphs_new = pdata.glyph(orient="B_unit", scale=False, factor=vector_scale)
            glyphs.copy_from(glyphs_new)
            glyphs_actor.mapper.dataset = glyphs

        plotter.add_text(
            (
                f"idx={state['idx']}  auto={state['auto_rotate']}  "
                f"az_speed={state['az_speed']:.1f}  line_motion={state['line_motion']}"
            ),
            position="upper_left",
            font_size=10,
            color="black",
            name="status_text",
        )
        plotter.render()

    plotter.show(auto_close=False, interactive_update=True)
    target_dt = max(0.005, interval_ms / 1000.0)
    while getattr(plotter, "iren", None) is not None:
        t0 = perf_counter()
        _tick()
        dt = perf_counter() - t0
        plotter.update(max(0.0, target_dt - dt))
    return plotter
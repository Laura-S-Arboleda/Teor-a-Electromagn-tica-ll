from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch

import physics

# ---------------------------------------------------------------------------
# Paleta e identidad visual del proyecto (coherente con la serie Maxwell)
# ---------------------------------------------------------------------------

COLOR_BG = "#0f1420"
COLOR_PANEL = "#161d2e"
COLOR_GRID = "#28324a"
COLOR_TEXT = "#dbe4f5"
COLOR_ACCENT_1 = "#4fd1c5"   # cian: flujo / bobina
COLOR_ACCENT_2 = "#f6ad55"   # naranja: FEM / imán
COLOR_ACCENT_3 = "#9f7aea"   # violeta: posición / fase
COLOR_ACCENT_4 = "#68d391"   # verde: campo eléctrico positivo
COLOR_ACCENT_5 = "#fc8181"   # rojo: campo eléctrico negativo / advertencias
COLOR_MARKER = "#ffffff"

plt_style = {
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_PANEL,
    "axes.edgecolor": COLOR_GRID,
    "axes.labelcolor": COLOR_TEXT,
    "axes.grid": True,
    "grid.color": COLOR_GRID,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "text.color": COLOR_TEXT,
    "font.size": 9,
    "font.family": "sans-serif",
    "legend.facecolor": COLOR_PANEL,
    "legend.edgecolor": COLOR_GRID,
    "legend.labelcolor": COLOR_TEXT,
}
matplotlib.rcParams.update(plt_style)


class BaseCanvas(FigureCanvasQTAgg):
    """Clase base para todos los canvases de la aplicación."""

    def __init__(self, width=5.0, height=3.6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=COLOR_BG)
        super().__init__(self.fig)
        self.fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.15)


# ---------------------------------------------------------------------------
# A. Animación del sistema físico (imán + bobina)
# ---------------------------------------------------------------------------

class SystemAnimationCanvas(BaseCanvas):
    """Esquema 2D del sistema: eje de movimiento, bobina (corte) e imán."""

    def __init__(self):
        super().__init__(width=5.6, height=3.8)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Sistema físico: imán en movimiento cerca de la bobina",
                           fontsize=10, color=COLOR_TEXT)
        self._init_static_elements()

    def _init_static_elements(self):
        ax = self.ax
        ax.set_xlim(-0.14, 0.14)
        ax.set_ylim(-0.06, 0.06)
        ax.set_aspect("equal")
        ax.set_xlabel("Posición axial z [m]")
        ax.set_yticks([])
        ax.axhline(0.0, color=COLOR_GRID, linewidth=1.0, zorder=1)

        # Bobina representada como dos "arrollamientos" (corte transversal)
        self.coil_top = Circle((0.0, 0.02), 0.006, facecolor="none",
                                edgecolor=COLOR_ACCENT_1, linewidth=2.0, zorder=3)
        self.coil_bottom = Circle((0.0, -0.02), 0.006, facecolor="none",
                                   edgecolor=COLOR_ACCENT_1, linewidth=2.0, zorder=3)
        ax.add_patch(self.coil_top)
        ax.add_patch(self.coil_bottom)
        ax.plot([0.0, 0.0], [-0.045, 0.045], color=COLOR_ACCENT_1,
                 linewidth=1.2, linestyle=":", alpha=0.6, zorder=2)
        ax.text(0.0, 0.05, "Bobina (eje z=0)", color=COLOR_ACCENT_1,
                 fontsize=8, ha="center")

        # Imán: rectángulo bicolor (N/S) representado por un marcador
        self.magnet_body, = ax.plot([], [], marker="s", markersize=22,
                                     color=COLOR_ACCENT_2, zorder=4)
        self.magnet_label = ax.text(0.0, -0.035, "", color=COLOR_ACCENT_2,
                                     fontsize=8, ha="center")
        self.velocity_arrow = None

    def update_scene(self, z_m: float, v_m: float, R_coil: float):
        """Actualiza la posición del imán y el tamaño visual de la bobina."""
        scale = min(0.02, max(0.008, R_coil * 0.6))
        self.coil_top.center = (0.0, scale)
        self.coil_bottom.center = (0.0, -scale)
        self.coil_top.set_radius(scale * 0.55)
        self.coil_bottom.set_radius(scale * 0.55)

        self.magnet_body.set_data([z_m], [0.0])
        self.magnet_label.set_position((z_m, -0.035))
        self.magnet_label.set_text(f"z = {z_m:+.3f} m")

        if self.velocity_arrow is not None:
            try:
                self.velocity_arrow.remove()
            except Exception:
                pass
            self.velocity_arrow = None

        if abs(v_m) > 1e-4:
            arrow_len = np.clip(v_m * 0.05, -0.03, 0.03)
            self.velocity_arrow = FancyArrowPatch(
                (z_m, 0.015), (z_m + arrow_len, 0.015),
                arrowstyle="-|>", mutation_scale=10,
                color=COLOR_MARKER, linewidth=1.4, zorder=5,
            )
            self.ax.add_patch(self.velocity_arrow)

        self.draw_idle()


# ---------------------------------------------------------------------------
# B/C. Flujo magnético y FEM en función del tiempo
# ---------------------------------------------------------------------------

class TimeSeriesCanvas(BaseCanvas):
    """Canvas genérico para una serie temporal con marcador de tiempo actual."""

    def __init__(self, title: str, ylabel: str, color: str):
        super().__init__(width=5.6, height=3.4)
        self.color = color
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title, fontsize=10, color=COLOR_TEXT)
        self.ax.set_xlabel("Tiempo t [s]")
        self.ax.set_ylabel(ylabel)
        self.line, = self.ax.plot([], [], color=color, linewidth=1.6)
        self.marker, = self.ax.plot([], [], marker="o", color=COLOR_MARKER,
                                     markersize=6, zorder=5)

    def set_data(self, t: np.ndarray, y: np.ndarray):
        self.line.set_data(t, y)
        if len(t) > 0:
            self.ax.set_xlim(t[0], t[-1])
            ymin, ymax = float(np.min(y)), float(np.max(y))
            pad = 0.1 * (ymax - ymin + 1e-12)
            self.ax.set_ylim(ymin - pad, ymax + pad)
        self.draw_idle()

    def set_marker(self, t_now: float, y_now: float):
        self.marker.set_data([t_now], [y_now])
        self.draw_idle()


# ---------------------------------------------------------------------------
# D. Relaciones de fase (posición, flujo, FEM normalizados)
# ---------------------------------------------------------------------------

class PhaseComparisonCanvas(BaseCanvas):
    """Comparación normalizada de posición, flujo y FEM para analizar fase."""

    def __init__(self):
        super().__init__(width=5.6, height=3.6)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Relaciones temporales y de fase (magnitudes normalizadas)",
                           fontsize=10, color=COLOR_TEXT)
        self.ax.set_xlabel("Tiempo t [s]")
        self.ax.set_ylabel("Magnitud normalizada [-1, 1]")
        self.ax.set_ylim(-1.15, 1.15)

        self.line_pos, = self.ax.plot([], [], color=COLOR_ACCENT_3,
                                       linewidth=1.5, label="Posición z_m(t)")
        self.line_flux, = self.ax.plot([], [], color=COLOR_ACCENT_1,
                                        linewidth=1.5, label="Flujo Φ_B(t)")
        self.line_emf, = self.ax.plot([], [], color=COLOR_ACCENT_2,
                                       linewidth=1.5, label="FEM ε(t)")
        self.marker_line = self.ax.axvline(0.0, color=COLOR_MARKER,
                                            linewidth=1.0, linestyle="--", alpha=0.7)
        self.ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    @staticmethod
    def _normalize(y: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(y))
        if peak < 1e-14:
            return np.zeros_like(y)
        return y / peak

    def set_data(self, t, z_m, phi_b, emf):
        self.line_pos.set_data(t, self._normalize(z_m))
        self.line_flux.set_data(t, self._normalize(phi_b))
        self.line_emf.set_data(t, self._normalize(emf))
        if len(t) > 0:
            self.ax.set_xlim(t[0], t[-1])
        self.draw_idle()

    def set_marker(self, t_now: float):
        self.marker_line.set_xdata([t_now, t_now])
        self.draw_idle()


# ---------------------------------------------------------------------------
# E. Campo eléctrico inducido: circulación espacial
# ---------------------------------------------------------------------------

class InducedEFieldCanvas(BaseCanvas):
    """Visualización de la circulación del campo eléctrico inducido E_phi(rho).

    Se representa, en el plano de la bobina (vista axial, mirando a lo largo
    de +z), un conjunto de lazos concéntricos con flechas tangenciales cuyo
    sentido y longitud codifican el signo y la magnitud de E_phi(rho),
    calculado en physics.induced_e_field_azimuthal a partir de la variación
    real del flujo encerrado (no es un dibujo arbitrario).
    """

    N_LOOPS = 5
    N_ARROWS_PER_LOOP = 10

    def __init__(self):
        super().__init__(width=6.2, height=6.2)
        self.fig.subplots_adjust(left=0.06, right=0.94, top=0.90, bottom=0.06)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Circulación espacial del campo E inducido\n(vista axial, plano de la bobina)",
                           fontsize=9.5, color=COLOR_TEXT)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.coil_circle = None
        self.arrows = []
        self.info_text = self.ax.text(
            0.02, 0.02, "", transform=self.ax.transAxes, fontsize=8,
            color=COLOR_TEXT, va="bottom", ha="left")

    def update_field(self, rho_values: np.ndarray, e_phi: np.ndarray,
                      R_coil: float, dphi_dt_now: float):
        ax = self.ax
        for a in self.arrows:
            a.remove()
        self.arrows = []
        if self.coil_circle is not None:
            self.coil_circle.remove()

        r_max = max(rho_values[-1], R_coil) * 1.15
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)

        self.coil_circle = Circle((0, 0), R_coil, facecolor="none",
                                   edgecolor=COLOR_ACCENT_1, linewidth=2.0,
                                   linestyle="-", zorder=2)
        ax.add_patch(self.coil_circle)

        e_max = np.max(np.abs(e_phi)) + 1e-30
        for loop_idx in range(self.N_LOOPS):
            frac = (loop_idx + 1) / self.N_LOOPS
            r_loop = frac * r_max * 0.92
            j = int(frac * (len(rho_values) - 1))
            e_val = e_phi[j]
            color = COLOR_ACCENT_4 if e_val >= 0 else COLOR_ACCENT_5
            arrow_scale = 0.35 + 0.65 * (abs(e_val) / e_max)

            for k in range(self.N_ARROWS_PER_LOOP):
                theta = 2.0 * np.pi * k / self.N_ARROWS_PER_LOOP
                x0 = r_loop * np.cos(theta)
                y0 = r_loop * np.sin(theta)
                # dirección tangencial; sentido según signo de e_val
                sense = 1.0 if e_val >= 0 else -1.0
                dtheta = sense * 0.18 * arrow_scale
                x1 = r_loop * np.cos(theta + dtheta)
                y1 = r_loop * np.sin(theta + dtheta)
                arrow = FancyArrowPatch((x0, y0), (x1, y1),
                                         arrowstyle="-|>", mutation_scale=6,
                                         color=color, linewidth=1.3,
                                         alpha=0.85, zorder=3)
                ax.add_patch(arrow)
                self.arrows.append(arrow)

        sentido = "antihorario" if dphi_dt_now < 0 else ("horario" if dphi_dt_now > 0 else "nulo")
        self.info_text.set_text(
            f"dΦ_B/dt = {dphi_dt_now:+.4e} Wb/s\n"
            f"Sentido de circulación de E: {sentido}\n"
            f"(convención: sentido opuesto al de dΦ_B/dt, Ley de Lenz)"
        )
        self.draw_idle()

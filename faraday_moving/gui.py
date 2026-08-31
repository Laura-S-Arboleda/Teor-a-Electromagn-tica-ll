from __future__ import annotations

import io
import csv
import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QGroupBox, QTabWidget,
    QScrollArea, QFrame, QFileDialog, QMessageBox, QSizePolicy, QTextEdit,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from physics import MagnetParams, CoilParams, magnetic_flux, induced_e_field_azimuthal
from solver import SimulationConfig, run_simulation, lenz_sign_check
import visualization as viz

APP_TITLE = "Ley de Inducción de Faraday"
APP_SUBTITLE = "Simulación numérica de un imán en movimiento cerca de una bobina"


# ---------------------------------------------------------------------------
# Utilidad: renderizar ecuaciones en formato matemático (mathtext) a imagen
# ---------------------------------------------------------------------------

def render_equation_pixmap(latex: str, fontsize: int = 18,
                            color: str = "#dbe4f5") -> QPixmap:
    """Renderiza una expresión mathtext de Matplotlib como QPixmap con fondo
    transparente, para mostrar ecuaciones con calidad tipográfica dentro de
    widgets Qt estándar (QLabel)."""
    fig = Figure(figsize=(6, 1.0), dpi=200)
    fig.patch.set_alpha(0.0)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.0, 0.5, latex, fontsize=fontsize, color=color,
            ha="left", va="center", transform=ax.transAxes)
    canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.05)
    buf.seek(0)
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue(), "PNG")
    return pixmap


def equation_label(latex: str, fontsize: int = 18, max_width: int = 520) -> QLabel:
    lbl = QLabel()
    pixmap = render_equation_pixmap(latex, fontsize=fontsize)
    if pixmap.width() > max_width:
        pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
    lbl.setPixmap(pixmap)
    lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return lbl


# ---------------------------------------------------------------------------
# Panel lateral de controles
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(340)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        layout.addWidget(self._build_magnet_group())
        layout.addWidget(self._build_motion_group())
        layout.addWidget(self._build_coil_group())
        layout.addWidget(self._build_numeric_group())
        layout.addWidget(self._build_buttons_group())
        layout.addStretch(1)

        scroll_wrap = QVBoxLayout()

    # -- Grupos de parámetros -------------------------------------------------

    def _spin(self, minimum, maximum, value, step, decimals=3):
        sb = QDoubleSpinBox()
        sb.setRange(minimum, maximum)
        sb.setValue(value)
        sb.setSingleStep(step)
        sb.setDecimals(decimals)
        return sb

    def _build_magnet_group(self) -> QGroupBox:
        box = QGroupBox("Parámetros del imán (dipolo)")
        grid = QGridLayout(box)

        self.spin_m = self._spin(0.1, 50.0, 8.0, 0.5, 2)
        grid.addWidget(QLabel("Momento dipolar m [A·m²]"), 0, 0)
        grid.addWidget(self.spin_m, 0, 1)

        self.spin_zc = self._spin(-0.10, 0.10, 0.0, 0.005, 3)
        grid.addWidget(QLabel("Posición central z_c [m]"), 1, 0)
        grid.addWidget(self.spin_zc, 1, 1)

        return box

    def _build_motion_group(self) -> QGroupBox:
        box = QGroupBox("Parámetros del movimiento oscilatorio")
        grid = QGridLayout(box)

        self.spin_amplitude = self._spin(0.005, 0.12, 0.08, 0.005, 3)
        grid.addWidget(QLabel("Amplitud A [m]"), 0, 0)
        grid.addWidget(self.spin_amplitude, 0, 1)

        self.spin_freq = self._spin(0.05, 5.0, 0.8, 0.05, 2)
        grid.addWidget(QLabel("Frecuencia f [Hz]"), 1, 0)
        grid.addWidget(self.spin_freq, 1, 1)

        self.spin_phase = self._spin(-180.0, 180.0, 0.0, 5.0, 1)
        grid.addWidget(QLabel("Fase inicial φ [°]"), 2, 0)
        grid.addWidget(self.spin_phase, 2, 1)

        return box

    def _build_coil_group(self) -> QGroupBox:
        box = QGroupBox("Parámetros de la bobina")
        grid = QGridLayout(box)

        self.spin_N = QSpinBox()
        self.spin_N.setRange(1, 5000)
        self.spin_N.setValue(300)
        grid.addWidget(QLabel("Número de espiras N"), 0, 0)
        grid.addWidget(self.spin_N, 0, 1)

        self.spin_R = self._spin(0.005, 0.10, 0.03, 0.001, 3)
        grid.addWidget(QLabel("Radio de la bobina R [m]"), 1, 0)
        grid.addWidget(self.spin_R, 1, 1)

        return box

    def _build_numeric_group(self) -> QGroupBox:
        box = QGroupBox("Parámetros numéricos")
        grid = QGridLayout(box)

        self.spin_tfinal = self._spin(0.5, 30.0, 5.0, 0.5, 2)
        grid.addWidget(QLabel("Duración t_final [s]"), 0, 0)
        grid.addWidget(self.spin_tfinal, 0, 1)

        self.spin_nsteps = QSpinBox()
        self.spin_nsteps.setRange(50, 20000)
        self.spin_nsteps.setValue(1000)
        self.spin_nsteps.setSingleStep(50)
        grid.addWidget(QLabel("Pasos temporales n"), 1, 0)
        grid.addWidget(self.spin_nsteps, 1, 1)

        self.lbl_dt = QLabel("Δt = -")
        grid.addWidget(self.lbl_dt, 2, 0, 1, 2)

        return box

    def _build_buttons_group(self) -> QGroupBox:
        box = QGroupBox("Control de la simulación")
        v = QVBoxLayout(box)

        self.btn_play_pause = QPushButton("▶ Iniciar")
        self.btn_reset = QPushButton("⟲ Reiniciar")
        self.btn_apply = QPushButton("Actualizar parámetros")
        self.btn_defaults = QPushButton("Restablecer valores por defecto")
        self.btn_export = QPushButton("Exportar resultados (CSV)")

        for b in (self.btn_play_pause, self.btn_reset, self.btn_apply,
                  self.btn_defaults, self.btn_export):
            v.addWidget(b)

        return box

    # -- Lectura / escritura de parámetros ------------------------------------

    def get_magnet_params(self) -> MagnetParams:
        return MagnetParams(
            m=self.spin_m.value(),
            z_c=self.spin_zc.value(),
            amplitude=self.spin_amplitude.value(),
            omega=2.0 * np.pi * self.spin_freq.value(),
            phase=np.deg2rad(self.spin_phase.value()),
            r_min=0.006,
        )

    def get_coil_params(self) -> CoilParams:
        return CoilParams(N_turns=self.spin_N.value(), R_coil=self.spin_R.value())

    def get_sim_config(self) -> SimulationConfig:
        return SimulationConfig(t_final=self.spin_tfinal.value(),
                                 n_steps=self.spin_nsteps.value())

    def set_defaults(self):
        self.spin_m.setValue(8.0)
        self.spin_zc.setValue(0.0)
        self.spin_amplitude.setValue(0.08)
        self.spin_freq.setValue(0.8)
        self.spin_phase.setValue(0.0)
        self.spin_N.setValue(300)
        self.spin_R.setValue(0.03)
        self.spin_tfinal.setValue(5.0)
        self.spin_nsteps.setValue(1000)

    def update_dt_label(self):
        t_final = self.spin_tfinal.value()
        n = self.spin_nsteps.value()
        dt = t_final / max(1, (n - 1))
        self.lbl_dt.setText(f"Δt = {dt:.5f} s")


# ---------------------------------------------------------------------------
# Panel de resultados numéricos
# ---------------------------------------------------------------------------

class NumericResultsPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Resultados numéricos instantáneos", parent)
        grid = QGridLayout(self)
        labels = [
            "Tiempo t", "Posición z_m(t)", "Velocidad v_m(t)",
            "Flujo Φ_B(t)", "dΦ_B/dt", "FEM ε(t)", "FEM máxima |ε|_max",
            "Espiras N", "Radio bobina R", "Δt",
        ]
        self.value_labels = {}
        for i, name in enumerate(labels):
            grid.addWidget(QLabel(name + ":"), i, 0)
            val_lbl = QLabel("-")
            val_lbl.setStyleSheet("font-weight: 600;")
            grid.addWidget(val_lbl, i, 1)
            self.value_labels[name] = val_lbl

    def update_values(self, t, z_m, v_m, phi, dphi_dt, emf, emf_max, N, R, dt):
        self.value_labels["Tiempo t"].setText(f"{t:.4f} s")
        self.value_labels["Posición z_m(t)"].setText(f"{z_m:+.5f} m")
        self.value_labels["Velocidad v_m(t)"].setText(f"{v_m:+.5f} m/s")
        self.value_labels["Flujo Φ_B(t)"].setText(f"{phi:+.6e} Wb")
        self.value_labels["dΦ_B/dt"].setText(f"{dphi_dt:+.6e} Wb/s")
        self.value_labels["FEM ε(t)"].setText(f"{emf:+.6e} V")
        self.value_labels["FEM máxima |ε|_max"].setText(f"{emf_max:.6e} V")
        self.value_labels["Espiras N"].setText(f"{N:d}")
        self.value_labels["Radio bobina R"].setText(f"{R:.4f} m")
        self.value_labels["Δt"].setText(f"{dt:.6f} s")


# ---------------------------------------------------------------------------
# Pestaña de fundamento teórico
# ---------------------------------------------------------------------------

def build_theory_tab() -> QWidget:
    widget = QWidget()
    outer = QVBoxLayout(widget)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    inner = QWidget()
    layout = QVBoxLayout(inner)
    layout.setSpacing(14)

    def add_section(title: str, latex: str, explanation: str):
        title_lbl = QLabel(title)
        f = QFont()
        f.setBold(True)
        f.setPointSize(11)
        title_lbl.setFont(f)
        layout.addWidget(title_lbl)
        layout.addWidget(equation_label(latex, fontsize=16))
        exp_lbl = QLabel(explanation)
        exp_lbl.setWordWrap(True)
        layout.addWidget(exp_lbl)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background-color: {viz.COLOR_GRID};")
        layout.addWidget(line)

    add_section(
        "Ley diferencial de Faraday",
        r"$\nabla\times\mathbf{E} = -\dfrac{\partial \mathbf{B}}{\partial t}$",
        "Un campo magnético que varía en el tiempo genera un campo eléctrico "
        "rotacional (no conservativo). Esta es la forma local de la ley."
    )
    add_section(
        "Forma integral",
        r"$\oint_C \mathbf{E}\cdot d\mathbf{l} = -\dfrac{d\Phi_B}{dt}$",
        "La circulación del campo eléctrico a lo largo de una curva cerrada C "
        "es igual a menos la tasa de cambio del flujo magnético que atraviesa "
        "cualquier superficie apoyada en C. Esta forma es la base física de la "
        "visualización de la circulación de E en la pestaña correspondiente."
    )
    add_section(
        "Flujo magnético",
        r"$\Phi_B = \int_S \mathbf{B}\cdot d\mathbf{A}$",
        "En esta simulación, B se calcula exactamente a partir del modelo de "
        "dipolo magnético del imán y se integra numéricamente (cuadratura de "
        "Simpson) sobre el área circular de la bobina."
    )
    add_section(
        "FEM inducida en una bobina de N espiras",
        r"$\mathcal{E} = -N\,\dfrac{d\Phi_B}{dt}$",
        "La FEM no se define arbitrariamente: se calcula tomando la derivada "
        "temporal NUMÉRICA del flujo magnético total (ya multiplicado por N), "
        "obtenido en cada instante a partir de la posición real del imán."
    )
    add_section(
        "Modelo cinemático del imán",
        r"$z_m(t) = z_c + A\cos(\omega t + \phi)$",
        "Movimiento oscilatorio sinusoidal a lo largo del eje de la bobina. "
        "La velocidad v_m(t) = dz_m/dt se obtiene analíticamente y se usa "
        "únicamente para la visualización del campo E inducido (regla de la "
        "cadena dB_z/dt = (dB_z/dz_m)·v_m), NO para definir la FEM."
    )
    add_section(
        "Aproximación de dipolo magnético",
        r"$\mathbf{B}(\mathbf{r}) = \dfrac{\mu_0}{4\pi r^3}\left[3(\hat{\mathbf{m}}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \hat{\mathbf{m}}\right]|\mathbf{m}|$",
        "El imán real se aproxima como un dipolo puntual. Esta aproximación es "
        "razonable cuando la distancia al punto de evaluación es grande frente "
        "al tamaño físico del imán, y deja de serlo cuando el imán está muy "
        "cerca de la bobina (por eso se introduce un suavizado numérico r_min)."
    )

    layout.addStretch(1)
    scroll.setWidget(inner)
    outer.addWidget(scroll)
    return widget


# ---------------------------------------------------------------------------
# Pestaña de análisis / validación numérica
# ---------------------------------------------------------------------------

def build_analysis_tab() -> QTextEdit:
    text = QTextEdit()
    text.setReadOnly(True)
    text.setMarkdown(
        "# Validación numérica y guía de interpretación\n\n"
        "Esta pestaña resume qué verificar al ejecutar la simulación (los "
        "valores concretos dependen de los parámetros elegidos y deben "
        "observarse directamente en las gráficas y el panel numérico):\n\n"
        "1. **Consistencia de signos (Ley de Lenz):** cuando el imán se "
        "aproxima a la bobina (Φ_B creciendo en magnitud), la FEM debe tener "
        "el signo que, según la Ley de Lenz, se opone al aumento del flujo.\n\n"
        "2. **FEM nula en los extremos del movimiento:** cerca de los puntos "
        "de retorno del oscilador (v_m ≈ 0), la FEM debe aproximarse a cero, "
        "incluso si el flujo en ese instante es grande.\n\n"
        "3. **Convergencia con la resolución temporal:** al aumentar el "
        "número de pasos temporales (n_steps) manteniendo t_final fijo, el "
        "valor máximo de |FEM| debe estabilizarse. Si cambia mucho al variar "
        "n_steps, la resolución actual es insuficiente.\n\n"
        "4. **Relación entre N y FEM:** al duplicar el número de espiras N, "
        "la FEM instantánea debe duplicarse aproximadamente, ya que "
        "ε = -N dΦ_B/dt (siendo Φ_B el flujo de una sola espira).\n\n"
        "5. **Errores por diferencias finitas:** con pocos pasos temporales, "
        "la derivada numérica (np.gradient) subestima los picos agudos de "
        "dΦ_B/dt. Aumentar n_steps reduce este efecto.\n\n"
        "Estas verificaciones deben repetirse experimentalmente cada vez que "
        "se cambian los parámetros; ningún valor numérico se ha fijado de "
        "antemano en este texto."
    )
    return text


# ---------------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 900)
        self.setStyleSheet(self._stylesheet())

        self.result = None
        self.frame_index = 0
        self.is_playing = False

        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._advance_frame)

        self._build_ui()
        self._connect_signals()
        self._run_new_simulation()

    # -- Construcción de la interfaz ------------------------------------------

    def _stylesheet(self) -> str:
        return f"""
        QMainWindow, QWidget {{ background-color: {viz.COLOR_BG}; color: {viz.COLOR_TEXT}; }}
        QGroupBox {{
            border: 1px solid {viz.COLOR_GRID}; border-radius: 6px; margin-top: 10px;
            font-weight: 600; padding-top: 8px;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
        QPushButton {{
            background-color: {viz.COLOR_PANEL}; border: 1px solid {viz.COLOR_GRID};
            border-radius: 4px; padding: 6px; color: {viz.COLOR_TEXT};
        }}
        QPushButton:hover {{ background-color: #1d2740; }}
        QPushButton:pressed {{ background-color: #24304f; }}
        QTabWidget::pane {{ border: 1px solid {viz.COLOR_GRID}; }}
        QTabBar::tab {{
            background: {viz.COLOR_PANEL}; color: {viz.COLOR_TEXT};
            padding: 8px 14px; border: 1px solid {viz.COLOR_GRID};
        }}
        QTabBar::tab:selected {{ background: #1d2740; }}
        QDoubleSpinBox, QSpinBox {{
            background-color: {viz.COLOR_PANEL}; border: 1px solid {viz.COLOR_GRID};
            border-radius: 3px; padding: 2px; color: {viz.COLOR_TEXT};
        }}
        QTextEdit {{ background-color: {viz.COLOR_PANEL}; border: none; }}
        QScrollArea {{ border: none; }}
        """

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(10, 10, 10, 10)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        self.control_panel = ControlPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.control_panel)
        scroll.setFixedWidth(360)
        body.addWidget(scroll)

        right_col = QVBoxLayout()
        self.tabs = QTabWidget()

        self.anim_canvas = viz.SystemAnimationCanvas()
        self.flux_canvas = viz.TimeSeriesCanvas(
            "Flujo magnético Φ_B(t)", "Φ_B [Wb]", viz.COLOR_ACCENT_1)
        self.emf_canvas = viz.TimeSeriesCanvas(
            "FEM inducida ε(t) = -N dΦ_B/dt", "ε [V]", viz.COLOR_ACCENT_2)
        self.phase_canvas = viz.PhaseComparisonCanvas()
        self.efield_canvas = viz.InducedEFieldCanvas()

        self.tabs.addTab(self.anim_canvas, "Sistema y animación")
        self.tabs.addTab(self.flux_canvas, "Flujo magnético")
        self.tabs.addTab(self.emf_canvas, "FEM inducida")
        self.tabs.addTab(self.phase_canvas, "Relaciones de fase")
        self.tabs.addTab(self.efield_canvas, "Campo E inducido")
        self.tabs.addTab(build_analysis_tab(), "Validación numérica")
        self.tabs.addTab(build_theory_tab(), "Fundamento teórico")

        right_col.addWidget(self.tabs, stretch=3)

        self.results_panel = NumericResultsPanel()
        right_col.addWidget(self.results_panel, stretch=1)

        body.addLayout(right_col, stretch=1)
        root.addLayout(body)

        self.status_label = QLabel("Listo.")
        self.status_label.setStyleSheet("color: #8fa3c9; font-size: 10px;")
        root.addWidget(self.status_label)

    def _build_header(self) -> QWidget:
        header = QWidget()
        v = QVBoxLayout(header)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)

        title = QLabel(APP_TITLE)
        f = QFont()
        f.setPointSize(20)
        f.setBold(True)
        title.setFont(f)

        subtitle = QLabel(APP_SUBTITLE)
        f2 = QFont()
        f2.setPointSize(11)
        subtitle.setFont(f2)
        subtitle.setStyleSheet("color: #9fb0d1;")

        eq = equation_label(r"$\nabla\times\mathbf{E} = -\dfrac{\partial \mathbf{B}}{\partial t}$",
                             fontsize=17)

        v.addWidget(title)
        v.addWidget(subtitle)
        v.addWidget(eq)
        return header

    def _connect_signals(self):
        cp = self.control_panel
        cp.btn_play_pause.clicked.connect(self._toggle_play)
        cp.btn_reset.clicked.connect(self._reset_simulation)
        cp.btn_apply.clicked.connect(self._run_new_simulation)
        cp.btn_defaults.clicked.connect(self._restore_defaults)
        cp.btn_export.clicked.connect(self._export_csv)
        cp.spin_tfinal.valueChanged.connect(cp.update_dt_label)
        cp.spin_nsteps.valueChanged.connect(cp.update_dt_label)
        cp.update_dt_label()

    # -- Lógica de simulación ---------------------------------------------------

    def _run_new_simulation(self):
        try:
            mag = self.control_panel.get_magnet_params()
            coil = self.control_panel.get_coil_params()
            cfg = self.control_panel.get_sim_config()
            self.result = run_simulation(mag, coil, cfg)
            assert lenz_sign_check(self.result), (
                "Inconsistencia interna detectada entre EMF y dPhi/dt.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error en la simulación", str(exc))
            return

        self.frame_index = 0
        self.flux_canvas.set_data(self.result.t, self.result.phi_b)
        self.emf_canvas.set_data(self.result.t, self.result.emf)
        self.phase_canvas.set_data(self.result.t, self.result.z_m,
                                    self.result.phi_b, self.result.emf)
        self.control_panel.update_dt_label()
        self._render_frame()
        self.status_label.setText(
            f"Simulación calculada: {cfg.n_steps} pasos, t_final={cfg.t_final:.2f} s. "
            f"Verificación de consistencia interna (EMF = -dΦ/dt): OK."
        )

    def _restore_defaults(self):
        self.control_panel.set_defaults()
        self._run_new_simulation()

    def _reset_simulation(self):
        self.is_playing = False
        self.timer.stop()
        self.control_panel.btn_play_pause.setText("▶ Iniciar")
        self.frame_index = 0
        self._render_frame()

    def _toggle_play(self):
        if self.result is None:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.control_panel.btn_play_pause.setText("⏸ Pausar")
            self.timer.start()
        else:
            self.control_panel.btn_play_pause.setText("▶ Iniciar")
            self.timer.stop()

    def _advance_frame(self):
        if self.result is None:
            return
        n = len(self.result.t)
        self.frame_index += max(1, n // 400)
        if self.frame_index >= n:
            self.frame_index = n - 1
            self.is_playing = False
            self.timer.stop()
            self.control_panel.btn_play_pause.setText("▶ Iniciar")
        self._render_frame()

    def _render_frame(self):
        if self.result is None:
            return
        res = self.result
        i = min(self.frame_index, len(res.t) - 1)

        t_now = res.t[i]
        z_now = res.z_m[i]
        v_now = res.v_m[i]
        phi_now = res.phi_b[i]
        dphi_now = res.dphi_dt[i]
        emf_now = res.emf[i]

        self.anim_canvas.update_scene(z_now, v_now, res.coil.R_coil)
        self.flux_canvas.set_marker(t_now, phi_now)
        self.emf_canvas.set_marker(t_now, emf_now)
        self.phase_canvas.set_marker(t_now)

        rho_values = np.linspace(0.15 * res.coil.R_coil, 2.2 * res.coil.R_coil, 24)
        e_phi = induced_e_field_azimuthal(rho_values, z_now, v_now, res.mag)
        self.efield_canvas.update_field(rho_values, e_phi, res.coil.R_coil, dphi_now)

        emf_max = float(np.max(np.abs(res.emf)))
        dt = res.t[1] - res.t[0] if len(res.t) > 1 else 0.0
        self.results_panel.update_values(
            t_now, z_now, v_now, phi_now, dphi_now, emf_now, emf_max,
            res.coil.N_turns, res.coil.R_coil, dt)

    # -- Exportación --------------------------------------------------------

    def _export_csv(self):
        if self.result is None:
            QMessageBox.warning(self, "Sin datos", "Ejecuta primero la simulación.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar resultados", "faraday_resultados.csv", "CSV (*.csv)")
        if not path:
            return
        res = self.result
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["t_s", "z_m_m", "v_m_m_s", "Phi_B_Wb",
                                  "dPhi_dt_Wb_s", "EMF_V"])
                for row in zip(res.t, res.z_m, res.v_m, res.phi_b,
                                res.dphi_dt, res.emf):
                    writer.writerow([f"{v:.8e}" for v in row])
            self.status_label.setText(f"Resultados exportados a: {path}")
        except OSError as exc:
            QMessageBox.critical(self, "Error al exportar", str(exc))


def create_application() -> tuple[QApplication, MainWindow]:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    return app, window

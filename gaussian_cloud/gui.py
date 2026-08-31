import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QTabWidget, QGroupBox,
    QFormLayout, QTextEdit, QMessageBox, QFileDialog, QFrame
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from physics import gaussian_charge_density, total_charge, EPS0_NORMALIZED
from solver import solve_poisson, compute_electric_field, compute_divergence, gauss_law_error
import visualization as viz


# Parámetros por defecto de la simulación (usados también por "Restablecer parámetros")
DEFAULT_PARAMS = {
    "rho0": 1.0,
    "sigma": 1.0,
    "x0": 0.0,
    "y0": 0.0,
    "L": 5.0,     # semi-ancho del dominio: dominio simulado en [-L, L] x [-L, L]
    "N": 121,     # número de nodos de malla por eje
}


class MplTab(QWidget):
    """Pestaña genérica que contiene una figura de Matplotlib embebida."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)


class TheoryTab(QWidget):
    """Pestaña de fundamento teórico: texto estático explicativo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet("font-size: 13px;")
        text.setHtml(self._html())
        layout.addWidget(text)

    @staticmethod
    def _html():
        return """
        <h2>Fundamento teórico</h2>

        <p><b>1. Ley de Gauss para la electricidad (forma diferencial):</b></p>
        <p style="font-size:16px;">&nabla; &middot; <b>E</b> = &rho; / &epsilon;<sub>0</sub></p>
        <p>Relaciona la divergencia del campo eléctrico en un punto con la densidad
        de carga (volumétrica en 3D, o superficial en este modelo 2D) presente en
        ese mismo punto.</p>

        <p><b>2. Relación entre potencial y campo eléctrico:</b></p>
        <p style="font-size:16px;"><b>E</b> = &minus;&nabla;V</p>
        <p>El campo eléctrico es el negativo del gradiente del potencial escalar V.</p>

        <p><b>3. Ecuación de Poisson (combinando 1 y 2):</b></p>
        <p style="font-size:16px;">&nabla;&sup2;V = &minus;&rho; / &epsilon;<sub>0</sub></p>
        <p>Esta es la ecuación que se resuelve numéricamente en esta aplicación mediante
        diferencias finitas y una matriz laplaciana dispersa (método directo, spsolve).</p>

        <p><b>4. Distribución de carga utilizada (nube gaussiana 2D):</b></p>
        <p style="font-size:16px;">&rho;(x,y) = &rho;<sub>0</sub>&nbsp;exp[ &minus;((x&minus;x<sub>0</sub>)&sup2;
        + (y&minus;y<sub>0</sub>)&sup2;) / (2&sigma;&sup2;) ]</p>
        <p>&rho;<sub>0</sub>: amplitud de la densidad en el centro de la nube.<br>
        &sigma;: ancho característico de la nube (mayor &sigma; &rarr; carga más dispersa).<br>
        (x<sub>0</sub>, y<sub>0</sub>): posición del centro de la nube.</p>

        <p><b>Nota sobre unidades:</b> esta simulación usa un sistema de unidades
        normalizado en el que &epsilon;<sub>0</sub> = 1, por conveniencia numérica y
        pedagógica, en lugar del valor real del Sistema Internacional
        (&epsilon;<sub>0</sub> &asymp; 8.85&times;10<sup>-12</sup> F/m). Esto no cambia el
        comportamiento cualitativo de los resultados, pero sí sus valores numéricos
        absolutos.</p>

        <p><b>Nota sobre el modelo 2D:</b> la nube de carga se modela en un plano (x, y).
        Esto representa exactamente una distribución de carga superficial 2D, o puede
        aproximarse a un corte de una nube 3D; en este último caso los resultados NO son
        idénticos a la solución tridimensional real, ya que el laplaciano en 2D y en 3D
        tienen soluciones fundamentales distintas (logarítmica vs. 1/r).</p>
        """


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Ley de Gauss para la Electricidad — Simulación de nube de carga gaussiana"
        )
        self.resize(1300, 820)

        self._last_state = None  # almacena el último resultado calculado

        self._build_ui()
        self._connect_signals()
        self.run_simulation()  # simulación inicial con parámetros por defecto

    # ---------------------------------------------------------------
    # Construcción de la interfaz
    # ---------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        main_layout.addWidget(self._build_header())

        body_layout = QHBoxLayout()
        body_layout.addWidget(self._build_control_panel(), stretch=0)
        body_layout.addWidget(self._build_results_area(), stretch=1)
        main_layout.addLayout(body_layout)

    def _build_header(self):
        box = QFrame()
        box.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(box)

        title = QLabel("Ley de Gauss para la Electricidad")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        subtitle = QLabel("Simulación numérica de una nube de carga gaussiana")
        subtitle.setStyleSheet("font-size: 13px; color: #555555;")
        equation = QLabel("∇ · E = ρ / ε₀        (Ley de Gauss, forma diferencial)")
        equation.setStyleSheet("font-size: 14px; font-style: italic;")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(equation)
        return box

    def _build_control_panel(self):
        params_group = QGroupBox("Parámetros de la simulación")
        params_group.setFixedWidth(300)
        form = QFormLayout()

        self.spin_rho0 = QDoubleSpinBox()
        self.spin_rho0.setRange(-50.0, 50.0)
        self.spin_rho0.setSingleStep(0.1)
        self.spin_rho0.setDecimals(3)
        self.spin_rho0.setValue(DEFAULT_PARAMS["rho0"])
        form.addRow("ρ₀ (amplitud de carga):", self.spin_rho0)

        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.05, 10.0)
        self.spin_sigma.setSingleStep(0.05)
        self.spin_sigma.setDecimals(3)
        self.spin_sigma.setValue(DEFAULT_PARAMS["sigma"])
        form.addRow("σ (ancho de la nube):", self.spin_sigma)

        self.spin_x0 = QDoubleSpinBox()
        self.spin_x0.setRange(-10.0, 10.0)
        self.spin_x0.setSingleStep(0.1)
        self.spin_x0.setValue(DEFAULT_PARAMS["x0"])
        form.addRow("x₀ (posición):", self.spin_x0)

        self.spin_y0 = QDoubleSpinBox()
        self.spin_y0.setRange(-10.0, 10.0)
        self.spin_y0.setSingleStep(0.1)
        self.spin_y0.setValue(DEFAULT_PARAMS["y0"])
        form.addRow("y₀ (posición):", self.spin_y0)

        self.spin_L = QDoubleSpinBox()
        self.spin_L.setRange(1.0, 20.0)
        self.spin_L.setSingleStep(0.5)
        self.spin_L.setValue(DEFAULT_PARAMS["L"])
        form.addRow("L (semi-ancho del dominio):", self.spin_L)

        self.spin_N = QSpinBox()
        self.spin_N.setRange(31, 241)
        self.spin_N.setSingleStep(10)
        self.spin_N.setValue(DEFAULT_PARAMS["N"])
        form.addRow("N (nodos por eje):", self.spin_N)

        params_group.setLayout(form)

        wrapper = QWidget()
        wlayout = QVBoxLayout(wrapper)
        wlayout.addWidget(params_group)

        btn_layout = QVBoxLayout()
        self.btn_update = QPushButton("Actualizar simulación")
        self.btn_reset = QPushButton("Restablecer parámetros")
        self.btn_export = QPushButton("Exportar resultados")
        for b in (self.btn_update, self.btn_reset, self.btn_export):
            b.setMinimumHeight(32)
        btn_layout.addWidget(self.btn_update)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addWidget(self.btn_export)
        wlayout.addLayout(btn_layout)

        wlayout.addWidget(self._build_results_panel())
        wlayout.addStretch()
        return wrapper

    def _build_results_panel(self):
        group = QGroupBox("Resultados numéricos")
        form = QFormLayout()

        self.lbl_Qtot = QLabel("—")
        self.lbl_rho_max = QLabel("—")
        self.lbl_E_max = QLabel("—")
        self.lbl_V_max = QLabel("—")
        self.lbl_V_rms = QLabel("—")
        self.lbl_err_rel = QLabel("—")
        self.lbl_err_abs = QLabel("—")
        self.lbl_dx = QLabel("—")

        form.addRow("Carga total integrada:", self.lbl_Qtot)
        form.addRow("ρ máx:", self.lbl_rho_max)
        form.addRow("|E| máx:", self.lbl_E_max)
        form.addRow("V máx:", self.lbl_V_max)
        form.addRow("V RMS:", self.lbl_V_rms)
        form.addRow("Error rel. (L2) Ley de Gauss:", self.lbl_err_rel)
        form.addRow("Error abs. máx Ley de Gauss:", self.lbl_err_abs)
        form.addRow("Δx = Δy:", self.lbl_dx)

        group.setLayout(form)
        return group

    def _build_results_area(self):
        self.tabs = QTabWidget()

        self.tab_charge = MplTab()
        self.tab_potential = MplTab()
        self.tab_field = MplTab()
        self.tab_gauss = MplTab()
        self.tab_theory = TheoryTab()

        self.tabs.addTab(self.tab_charge, "A. Densidad de carga")
        self.tabs.addTab(self.tab_potential, "B. Potencial eléctrico")
        self.tabs.addTab(self.tab_field, "C. Campo y equipotenciales")
        self.tabs.addTab(self.tab_gauss, "D. Verificación Ley de Gauss")
        self.tabs.addTab(self.tab_theory, "Fundamento teórico")

        return self.tabs

    # ---------------------------------------------------------------
    # Conexión de señales
    # ---------------------------------------------------------------
    def _connect_signals(self):
        self.btn_update.clicked.connect(self.run_simulation)
        self.btn_reset.clicked.connect(self.reset_parameters)
        self.btn_export.clicked.connect(self.export_results)

    # ---------------------------------------------------------------
    # Lógica de simulación
    # ---------------------------------------------------------------
    def reset_parameters(self):
        """Restablece los controles a los valores por defecto y recalcula."""
        self.spin_rho0.setValue(DEFAULT_PARAMS["rho0"])
        self.spin_sigma.setValue(DEFAULT_PARAMS["sigma"])
        self.spin_x0.setValue(DEFAULT_PARAMS["x0"])
        self.spin_y0.setValue(DEFAULT_PARAMS["y0"])
        self.spin_L.setValue(DEFAULT_PARAMS["L"])
        self.spin_N.setValue(DEFAULT_PARAMS["N"])
        self.run_simulation()

    def run_simulation(self):
        """
        Lee los parámetros actuales de la interfaz, resuelve la simulación
        completa (carga -> potencial -> campo -> divergencia -> error) y
        actualiza todas las gráficas y el panel de resultados numéricos.
        """
        try:
            rho0 = self.spin_rho0.value()
            sigma = self.spin_sigma.value()
            x0 = self.spin_x0.value()
            y0 = self.spin_y0.value()
            L = self.spin_L.value()
            N = int(self.spin_N.value())

            if sigma <= 0:
                raise ValueError("σ debe ser mayor que cero.")
            if L <= 0:
                raise ValueError("El tamaño del dominio (L) debe ser mayor que cero.")
            if sigma > L:
                QMessageBox.warning(
                    self, "Advertencia física",
                    "σ es mayor que el semi-ancho del dominio L.\n"
                    "La condición de frontera V=0 puede dejar de ser una buena "
                    "aproximación, y el error numérico de la Ley de Gauss puede "
                    "aumentar cerca del borde del dominio."
                )

            x = np.linspace(-L, L, N)
            y = np.linspace(-L, L, N)
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            X, Y = np.meshgrid(x, y, indexing="ij")

            eps0 = EPS0_NORMALIZED

            rho = gaussian_charge_density(X, Y, rho0, sigma, x0, y0)
            V = solve_poisson(rho, dx, dy, eps0)
            Ex, Ey = compute_electric_field(V, dx, dy)
            divE = compute_divergence(Ex, Ey, dx, dy)
            gauss = gauss_law_error(divE, rho, eps0)

            self._last_state = dict(
                X=X, Y=Y, rho=rho, V=V, Ex=Ex, Ey=Ey, divE=divE,
                gauss=gauss, dx=dx, dy=dy,
            )

            viz.plot_charge_density(self.tab_charge.figure, X, Y, rho)
            self.tab_charge.canvas.draw()

            viz.plot_potential(self.tab_potential.figure, X, Y, V)
            self.tab_potential.canvas.draw()

            viz.plot_field_and_equipotentials(self.tab_field.figure, X, Y, V, Ex, Ey)
            self.tab_field.canvas.draw()

            viz.plot_gauss_verification(
                self.tab_gauss.figure, X, Y, divE, gauss["target"], gauss["error_map"]
            )
            self.tab_gauss.canvas.draw()

            self._update_results_panel(rho, V, Ex, Ey, gauss, dx, dy)

        except Exception as exc:  # manejo básico de errores de parámetros/numéricos
            QMessageBox.critical(self, "Error en la simulación", f"Ocurrió un error:\n{exc}")

    def _update_results_panel(self, rho, V, Ex, Ey, gauss, dx, dy):
        Qtot = total_charge(rho, dx, dy)
        Emag = np.sqrt(Ex ** 2 + Ey ** 2)

        self.lbl_Qtot.setText(f"{Qtot:.4f}")
        self.lbl_rho_max.setText(f"{np.max(rho):.4f}")
        self.lbl_E_max.setText(f"{np.max(Emag):.4f}")
        self.lbl_V_max.setText(f"{np.max(V):.4f}")
        self.lbl_V_rms.setText(f"{np.sqrt(np.mean(V ** 2)):.4f}")

        rel = gauss["relative_l2_error"]
        self.lbl_err_rel.setText("N/A" if np.isnan(rel) else f"{rel:.4%}")
        self.lbl_err_abs.setText(f"{gauss['max_abs_error']:.4e}")
        self.lbl_dx.setText(f"{dx:.4f}")

    def export_results(self):
        """Exporta el último estado calculado a un archivo .npz (NumPy)."""
        if self._last_state is None:
            QMessageBox.information(self, "Exportar", "Ejecuta primero una simulación.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar resultados", "resultados_gauss.npz", "Archivos NumPy (*.npz)"
        )
        if not path:
            return

        state = self._last_state
        try:
            np.savez(
                path,
                X=state["X"], Y=state["Y"], rho=state["rho"], V=state["V"],
                Ex=state["Ex"], Ey=state["Ey"], divE=state["divE"],
                target=state["gauss"]["target"], error_map=state["gauss"]["error_map"],
            )
            QMessageBox.information(self, "Exportar", f"Resultados exportados a:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error al exportar", f"No se pudo exportar:\n{exc}")


def create_app() -> QApplication:
    """Crea (o recupera) la instancia de QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

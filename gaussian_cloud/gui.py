import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QTabWidget, QGroupBox,
    QFormLayout, QTextEdit, QMessageBox, QFileDialog, QFrame
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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


class Density3DTab(QWidget):
    """Superficie 3D de la densidad de carga: la 'montaña' gaussiana."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6), dpi=120)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setVisible(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def plot(self, X, Y, Z, field_name="rho", field_label=None, title=None):
        """
        Grafica cualquier campo escalar 2D como superficie 3D.

        field_name: "rho" para densidad de carga, "V" para potencial eléctrico.
        field_label / title: si no se especifican, se infieren de field_name.
        """
        if field_label is None:
            field_label = "ρ(x,y)" if field_name == "rho" else "V(x,y)"
        if title is None:
            title = (
                "Densidad de carga 3D — montaña gaussiana"
                if field_name == "rho"
                else "Potencial eléctrico 3D — relieve de V(x,y)"
            )

        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")

        # Superficie densa para que la montaña se vea suave incluso al hacer zoom.
        surf = ax.plot_surface(
            X, Y, Z,
            cmap="viridis",
            rcount=min(180, X.shape[0]),
            ccount=min(180, X.shape[1]),
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        # Proyección del plano z=0 para visualizar el soporte espacial.
        z0 = np.zeros_like(Z)
        ax.plot_surface(
            X, Y, z0,
            color="lightgray",
            alpha=0.08,
            linewidth=0,
            antialiased=True,
        )

        ax.set_title(title, fontsize=14, pad=14)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(field_label)
        ax.view_init(elev=28, azim=-55)

        # Mantener proporciones razonables y permitir zoom/rotación con la barra.
        ax.set_box_aspect((1, 1, 0.65))

        cbar = self.figure.colorbar(surf, ax=ax, shrink=0.72, pad=0.10)
        cbar.set_label("Densidad de carga ρ" if field_name == "rho" else "Potencial eléctrico V")

        self.figure.tight_layout()
        self.canvas.draw()


class Gaussian3DTab(QWidget):
    """Geometría espacial de una nube gaussiana 3D."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6), dpi=120)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def plot(self, rho0, sigma, x0, y0, L):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")

        extent = min(L, max(3.2 * sigma, 0.85 * L))
        rho_max = max(abs(rho0), 1e-12)
        signo_val = 1.0 if rho0 >= 0 else -1.0

        # Como la nube es gaussiana isotrópica, rho(r) = rho0 * exp(-r^2/(2*sigma^2)),
        # las superficies de densidad constante son ESFERAS EXACTAS.
        # El radio de cada nivel se obtiene analíticamente (sin marching cubes,
        # sin muestreo de puntos, y por tanto sin artefactos de aliasing/moiré):
        #     frac = exp(-r^2/(2*sigma^2))  =>  r = sigma * sqrt(-2*ln(frac))
        levels = (0.70, 0.45, 0.25, 0.10)
        # Más transparentes en general para poder ver el centro a través de ellas.
        alphas = (0.32, 0.22, 0.15, 0.09)

        # Malla angular para cada esfera (independiente de la malla espacial del dominio).
        n_theta, n_phi = 60, 30
        theta = np.linspace(0, 2 * np.pi, n_theta)
        phi = np.linspace(0, np.pi, n_phi)
        theta, phi = np.meshgrid(theta, phi)

        cmap = plt.get_cmap("coolwarm")
        frac_min, frac_max = min(levels), max(levels)

        # Calculamos radio de cada nivel y ordenamos de mayor a menor radio,
        # para dibujar primero las capas exteriores (más grandes, más
        # transparentes) y al final las interiores (más pequeñas, más densas),
        # de modo que el centro quede visible por encima de todas.
        level_data = []
        for frac, alpha in zip(levels, alphas):
            if frac >= 1.0:
                continue
            r_level = sigma * np.sqrt(-2.0 * np.log(frac))
            if r_level > extent:
                continue  # nivel fuera del recorte visible, se omite
            level_data.append((frac, alpha, r_level))
        level_data.sort(key=lambda item: item[2], reverse=True)  # radio: mayor -> menor

        for frac, alpha, r_level in level_data:
            Xs = x0 + r_level * np.sin(phi) * np.cos(theta)
            Ys = y0 + r_level * np.sin(phi) * np.sin(theta)
            Zs = r_level * np.cos(phi)

            # Color estirado sobre un rango amplio del colormap (evita que
            # todos los niveles caigan en tonos muy parecidos): niveles bajos
            # de densidad (frac cerca de frac_min) quedan en tonos claros y
            # niveles altos (frac cerca de frac_max) en tonos intensos.
            t = (frac - frac_min) / (frac_max - frac_min)  # 0 (nivel externo) a 1 (interno)
            color_val = 0.5 + signo_val * (0.08 + 0.42 * t)
            color = cmap(color_val)

            ax.plot_surface(
                Xs, Ys, Zs,
                color=color,
                alpha=alpha,
                linewidth=0,
                antialiased=True,
                shade=True,
            )

        # Puntos de carga dentro de la nube ("puntitos"): se muestrean
        # con densidad de probabilidad radial proporcional a rho(r), de modo
        # que se concentran naturalmente cerca del centro sin necesidad de
        # filtrar una malla (evita el artefacto de moiré visto anteriormente).
        n_points = 900
        rng = np.random.default_rng(7)

        # Para una gaussiana 3D, el radio se puede muestrear invirtiendo su CDF
        # radial: r = sigma * sqrt(-2 ln(1 - u)) truncado al recorte visible.
        u = rng.uniform(0.0, 1.0 - 0.10, size=n_points)  # excluye la cola > extent aprox.
        r_samples = sigma * np.sqrt(-2.0 * np.log(1.0 - u))
        r_samples = r_samples[r_samples <= extent]

        n_valid = len(r_samples)
        costheta = rng.uniform(-1.0, 1.0, size=n_valid)
        phi_s = np.arccos(costheta)
        theta_s = rng.uniform(0.0, 2 * np.pi, size=n_valid)

        Xp = x0 + r_samples * np.sin(phi_s) * np.cos(theta_s)
        Yp = y0 + r_samples * np.sin(phi_s) * np.sin(theta_s)
        Zp = r_samples * np.cos(phi_s)

        rho_at_points = rho0 * np.exp(-r_samples**2 / (2.0 * sigma**2))

        ax.scatter(
            Xp, Yp, Zp,
            c=rho_at_points,
            cmap="coolwarm",
            vmin=-rho_max, vmax=rho_max,
            s=6,
            alpha=0.55,
            linewidths=0,
            depthshade=True,
        )

        # Centro de la nube.
        ax.scatter(
            [x0], [y0], [0],
            s=70,
            c="black",
            edgecolors="white",
            linewidths=1.0,
            depthshade=True,
        )
        ax.text(x0, y0, 0, "  centro", fontsize=9)

        ax.set_title("Geometría 3D de una nube de carga gaussiana", fontsize=14, pad=14)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(x0 - extent, x0 + extent)
        ax.set_ylim(y0 - extent, y0 + extent)
        ax.set_zlim(-extent, extent)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-55)

        signo = "+" if rho0 >= 0 else "−"
        info_text = (
            f"$\\rho_0$ = {rho0:.3f}\n"
            f"$\\sigma$ = {sigma:.3f}\n"
            f"signo = {signo}"
        )
        ax.text2D(
            0.02, 0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="gray",
                alpha=0.85,
            ),
        )

        self.figure.tight_layout()
        self.canvas.draw()



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
        self.tab_density_3d = Density3DTab()
        self.tab_gaussian_3d = Gaussian3DTab()
        self.tab_theory = TheoryTab()

        self.tabs.addTab(self.tab_charge, "A. Densidad de carga")
        self.tabs.addTab(self.tab_potential, "B. Potencial eléctrico")
        self.tabs.addTab(self.tab_field, "C. Campo y equipotenciales")
        self.tabs.addTab(self.tab_gauss, "D. Verificación Ley de Gauss")
        self.tabs.addTab(self.tab_density_3d, "E. Densidad 3D — montaña")
        self.tabs.addTab(self.tab_gaussian_3d, "F. Nube gaussiana 3D")
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

            # Montaña 3D de la densidad de carga.
            self.tab_density_3d.plot(X, Y, rho, field_name="rho")

            viz.plot_potential(self.tab_potential.figure, X, Y, V)
            self.tab_potential.canvas.draw()

            viz.plot_field_and_equipotentials(self.tab_field.figure, X, Y, V, Ex, Ey)
            self.tab_field.canvas.draw()

            viz.plot_gauss_verification(
                self.tab_gauss.figure, X, Y, divE, gauss["target"], gauss["error_map"]
            )
            self.tab_gauss.canvas.draw()

            self.tab_gaussian_3d.plot(
                rho0=rho0,
                sigma=sigma,
                x0=x0,
                y0=y0,
                L=L,
            )

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

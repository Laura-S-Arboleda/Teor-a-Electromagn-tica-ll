import numpy as np
from matplotlib.figure import Figure


def _style_axes(ax, title, xlabel="x (u.l.)", ylabel="y (u.l.)"):
    """Aplica un estilo consistente (título, etiquetas, proporciones) a un eje."""
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal", adjustable="box")


def plot_charge_density(fig: Figure, X, Y, rho):
    """Mapa de color de la densidad de carga rho(x,y)."""
    fig.clear()
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(X, Y, rho, shading="auto", cmap="RdBu_r")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\rho(x,y)$  (u. de carga / u.l.$^2$)", fontsize=9)
    _style_axes(ax, "Distribución de densidad de carga")
    fig.tight_layout()


def plot_potential(fig: Figure, X, Y, V):
    """Mapa de color del potencial eléctrico V(x,y)."""
    fig.clear()
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(X, Y, V, shading="auto", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$V(x,y)$  (u. de potencial)", fontsize=9)
    _style_axes(ax, "Potencial eléctrico")
    fig.tight_layout()


def plot_field_and_equipotentials(fig: Figure, X, Y, V, Ex, Ey):
    """
    Muestra simultáneamente:
      - Curvas equipotenciales (contornos de V).
      - Vectores del campo eléctrico normalizados en dirección, coloreados
        según su magnitud (para no saturar visualmente cerca del centro de
        la nube, donde el campo puede ser mucho más intenso que en la
        periferia).
    """
    fig.clear()
    ax = fig.add_subplot(111)

    cs = ax.contour(X, Y, V, levels=12, colors="black", linewidths=0.6, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")

    nx, ny = X.shape
    step = max(1, nx // 18)
    Emag = np.sqrt(Ex ** 2 + Ey ** 2)
    Emag_safe = np.where(Emag == 0, 1.0, Emag)  # evita división por cero al normalizar dirección

    q = ax.quiver(
        X[::step, ::step], Y[::step, ::step],
        (Ex / Emag_safe)[::step, ::step], (Ey / Emag_safe)[::step, ::step],
        Emag[::step, ::step], cmap="inferno", scale=25, width=0.004,
    )
    cbar = fig.colorbar(q, ax=ax)
    cbar.set_label(r"$|\mathbf{E}|$ (u. de campo)", fontsize=9)
    _style_axes(ax, "Campo eléctrico y curvas equipotenciales")
    fig.tight_layout()


def plot_gauss_verification(fig: Figure, X, Y, divE, target, error_map):
    """
    Muestra tres paneles lado a lado:
      1. Divergencia numérica del campo eléctrico.
      2. Valor teórico esperado rho/eps0.
      3. Error absoluto puntual entre ambos.

    Los dos primeros paneles comparten la misma escala de color (simétrica
    respecto a cero) para facilitar la comparación visual directa.
    """
    fig.clear()
    axs = fig.subplots(1, 3)

    vmax = max(np.max(np.abs(divE)), np.max(np.abs(target)), 1e-30)

    im0 = axs[0].pcolormesh(X, Y, divE, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im0, ax=axs[0], fraction=0.046)
    _style_axes(axs[0], r"$\nabla\cdot\mathbf{E}$ (numérico)")

    im1 = axs[1].pcolormesh(X, Y, target, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im1, ax=axs[1], fraction=0.046)
    _style_axes(axs[1], r"$\rho/\varepsilon_0$ (teórico)")

    im2 = axs[2].pcolormesh(X, Y, error_map, shading="auto", cmap="magma")
    fig.colorbar(im2, ax=axs[2], fraction=0.046)
    _style_axes(axs[2], "Error absoluto puntual")

    fig.tight_layout()

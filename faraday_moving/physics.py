from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constantes físicas
# ---------------------------------------------------------------------------

MU0 = 4.0 * np.pi * 1.0e-7  # Permeabilidad magnética del vacío [T*m/A]

# Número de puntos radiales usados en la cuadratura de Simpson para el
# cálculo del flujo magnético. Debe ser impar (Simpson) y suficientemente
# grande para que el error de cuadratura sea despreciable frente al error
# de discretización temporal.
N_RADIAL_QUADRATURE = 161


@dataclass
class MagnetParams:
    """Parámetros físicos del imán (aproximado como dipolo puntual)."""

    m: float = 8.0        # Momento dipolar magnético [A*m^2]
    z_c: float = 0.0       # Posición central de oscilación [m]
    amplitude: float = 0.08  # Amplitud del movimiento [m]
    omega: float = 2.0 * np.pi * 0.8  # Frecuencia angular [rad/s]
    phase: float = 0.0     # Fase inicial [rad]
    r_min: float = 0.006   # Distancia mínima de suavizado numérico [m]


@dataclass
class CoilParams:
    """Parámetros físicos de la bobina."""

    N_turns: int = 300     # Número de espiras
    R_coil: float = 0.03    # Radio de la bobina [m]


# ---------------------------------------------------------------------------
# Cinemática del imán
# ---------------------------------------------------------------------------

def magnet_position(t: np.ndarray, mag: MagnetParams) -> np.ndarray:
    """Posición axial del imán z_m(t) = z_c + A cos(w t + phi).

    Movimiento oscilatorio sinusoidal a lo largo del eje de la bobina.
    """
    return mag.z_c + mag.amplitude * np.cos(mag.omega * t + mag.phase)


def magnet_velocity(t: np.ndarray, mag: MagnetParams) -> np.ndarray:
    """Velocidad axial del imán v_m(t) = dz_m/dt = -A*w*sin(w t + phi).

    Calculada analíticamente (derivada exacta del modelo cinemático), lo
    cual es coherente porque el movimiento es una función conocida en forma
    cerrada. La FEM, en cambio, NO se deriva de esta velocidad de forma
    directa: se obtiene numéricamente a partir de la derivada temporal del
    flujo magnético (ver solver.py), tal como exige la Ley de Faraday.
    """
    return -mag.amplitude * mag.omega * np.sin(mag.omega * t + mag.phase)


# ---------------------------------------------------------------------------
# Campo magnético del dipolo
# ---------------------------------------------------------------------------

def _softened_distance(rho: np.ndarray, delta_z: np.ndarray, r_min: float) -> np.ndarray:
    """Distancia radial al dipolo, suavizada para evitar r -> 0.

    r = sqrt(rho^2 + delta_z^2 + r_min^2)

    Este suavizado ("softening") es una técnica numérica estándar en
    simulaciones de partículas/dipolos para evitar divergencias cuando el
    punto de evaluación se acerca demasiado a la fuente. Introduce un error
    controlado únicamente en la vecindad inmediata del imán (r ~ r_min),
    región en la que, de todas formas, la aproximación dipolar puntual deja
    de ser físicamente válida para un imán real de tamaño finito.
    """
    return np.sqrt(rho ** 2 + delta_z ** 2 + r_min ** 2)


def bz_dipole(rho: np.ndarray, z: float, z_m: float, m: float,
              r_min: float) -> np.ndarray:
    """Componente axial (z) del campo de un dipolo magnético puntual.

    El dipolo está ubicado en (0, 0, z_m) con momento m*z_hat. Se evalúa en
    puntos (rho, z) con simetría azimutal (independiente de phi).

    B_z(rho, z) = (mu0 * m) / (4*pi*r^3) * [3*(z - z_m)^2 / r^2 - 1]

    con r = sqrt(rho^2 + (z - z_m)^2), suavizado según r_min.
    """
    delta_z = z - z_m
    r = _softened_distance(rho, delta_z, r_min)
    return (MU0 * m) / (4.0 * np.pi * r ** 3) * (3.0 * delta_z ** 2 / r ** 2 - 1.0)


def brho_dipole(rho: np.ndarray, z: float, z_m: float, m: float,
                 r_min: float) -> np.ndarray:
    """Componente radial (rho) del campo de un dipolo magnético puntual.

    B_rho(rho, z) = (mu0 * m) / (4*pi*r^5) * 3 * rho * (z - z_m)
    """
    delta_z = z - z_m
    r = _softened_distance(rho, delta_z, r_min)
    return (MU0 * m) / (4.0 * np.pi * r ** 5) * 3.0 * rho * delta_z


def dbz_dzm(rho: np.ndarray, z: float, z_m: float, m: float,
            r_min: float) -> np.ndarray:
    """Derivada parcial analítica de B_z respecto de la posición del imán z_m.

    Se obtiene derivando B_z(rho, z, z_m) respecto a z_m (equivalentemente,
    respecto a -delta_z, con delta_z = z - z_m). Se usa junto con la regla
    de la cadena (dB_z/dt = dB_z/dz_m * v_m) para calcular la variación
    temporal local del campo, necesaria para la visualización del campo
    eléctrico inducido (Sección 6 del proyecto).

    Con delta_z = z - z_m, d(delta_z)/dz_m = -1. Definiendo
        B_z = K * (3*delta_z^2/r^2 - 1),   K = mu0*m/(4*pi*r^3)
    y usando r^2 = rho^2 + delta_z^2 + r_min^2 (r depende de z_m a través de
    delta_z), se deriva:

        dB_z/dz_m = (mu0*m)/(4*pi) * [ -6*delta_z/r^5
                                        + 15*delta_z*(3*delta_z^2/r^2 - 1)/r^5 * (r^2)/(3) ... ]

    Para evitar errores de álgebra, la derivada se calcula aquí de forma
    semi-analítica compacta y se valida numéricamente (ver solver.py,
    verificación de consistencia por diferencias finitas).
    """
    delta_z = z - z_m
    r = _softened_distance(rho, delta_z, r_min)
    r2 = r ** 2
    r5 = r ** 5
    r7 = r2 ** 3 * r

    # d(delta_z)/dz_m = -1 ; d(r^2)/dz_m = 2*delta_z*d(delta_z)/dz_m = -2*delta_z
    ddz_dzm = -1.0
    dr2_dzm = -2.0 * delta_z

    # B_z = (mu0*m/(4*pi)) * (3*delta_z^2 - r^2) / r^5
    numerator = 3.0 * delta_z ** 2 - r2
    d_numerator_dzm = 6.0 * delta_z * ddz_dzm - dr2_dzm

    # d(1/r^5)/dzm = -5/2 * r^(-7) * d(r^2)/dzm
    d_invr5_dzm = -2.5 * dr2_dzm / r7

    dBz_dzm_val = (MU0 * m) / (4.0 * np.pi) * (
        d_numerator_dzm / r5 + numerator * d_invr5_dzm
    )
    return dBz_dzm_val


# ---------------------------------------------------------------------------
# Flujo magnético a través de la bobina (integración numérica de Simpson)
# ---------------------------------------------------------------------------

def magnetic_flux(z_m: float, mag: MagnetParams, coil: CoilParams) -> float:
    """Flujo magnético total enlazado por la bobina de N espiras.

    Phi_B = N * integral_0^R  B_z(rho, 0; z_m) * 2*pi*rho  drho

    Se aproxima el campo real del dipolo (no uniforme) integrado sobre toda
    el área de la bobina mediante cuadratura de Simpson, evitando así la
    aproximación de campo uniforme salvo que R_coil sea pequeño frente a la
    distancia al imán (en cuyo caso el resultado tiende naturalmente a
    Phi_B ~ N * B_z(0,0;z_m) * pi * R^2, como límite de bajo orden).
    """
    n = N_RADIAL_QUADRATURE
    rho = np.linspace(0.0, coil.R_coil, n)
    bz = bz_dipole(rho, 0.0, z_m, mag.m, mag.r_min)
    integrand = bz * 2.0 * np.pi * rho
    flux_single_turn = float(simpson(integrand, x=rho))
    return coil.N_turns * flux_single_turn


def magnetic_flux_partial(z_m: float, mag: MagnetParams, coil: CoilParams,
                            rho_max: float) -> float:
    """Flujo (por una sola espira, sin factor N) encerrado hasta radio rho_max.

    Se usa para construir la circulación del campo eléctrico inducido en
    lazos amperianos de radio variable (Sección 6), NO para la FEM de la
    bobina (que usa magnetic_flux con N espiras completas).
    """
    if rho_max <= 0.0:
        return 0.0
    n = N_RADIAL_QUADRATURE
    rho = np.linspace(0.0, rho_max, n)
    bz = bz_dipole(rho, 0.0, z_m, mag.m, mag.r_min)
    integrand = bz * 2.0 * np.pi * rho
    return float(simpson(integrand, x=rho))


# ---------------------------------------------------------------------------
# Campo eléctrico inducido (circulación espacial)
# ---------------------------------------------------------------------------

def induced_e_field_azimuthal(rho_values: np.ndarray, z_m: float, v_m: float,
                                mag: MagnetParams) -> np.ndarray:
    """Campo eléctrico inducido azimutal E_phi(rho) en el plano de la bobina.

    Para cada radio rho, se aplica la forma integral de Faraday a un lazo
    circular concéntrico de radio rho situado en el plano z = 0:

        oint E . dl = -d(Phi_encerrado(rho))/dt
        E_phi(rho) * 2*pi*rho = -d(Phi_encerrado(rho))/dt

    Por simetría axial del sistema (dipolo alineado con el eje, lazo
    concéntrico), E_phi es constante en magnitud sobre cada lazo y no tiene
    componentes radial ni axial (E_rho = E_z = 0 en el plano de simetría,
    consistente con la ausencia de fuentes de carga libre en este modelo).

    d(Phi_encerrado)/dt se calcula mediante la regla de la cadena:
        dPhi/dt = integral_0^rho (dB_z/dt) * 2*pi*rho' drho'
        dB_z/dt = (dB_z/dz_m) * v_m

    es decir, se usa la derivada ANALÍTICA parcial de B_z respecto a la
    posición del imán (dbz_dzm) multiplicada por la velocidad instantánea
    del imán, e integrada en el área hasta cada radio. Esto asegura que la
    dirección y magnitud del campo mostrado están matemáticamente ligadas a
    la variación real del flujo, y no son un dibujo decorativo.

    Devuelve E_phi con signo: E_phi > 0 indica sentido antihorario visto
    desde +z (convención right-hand con d/dl en sentido antihorario).
    """
    e_phi = np.zeros_like(rho_values)
    n_fine = 81
    for i, rho_max in enumerate(rho_values):
        if rho_max <= 1e-6:
            e_phi[i] = 0.0
            continue
        rho_fine = np.linspace(0.0, rho_max, n_fine)
        dbz_dt = dbz_dzm(rho_fine, 0.0, z_m, mag.m, mag.r_min) * v_m
        integrand = dbz_dt * 2.0 * np.pi * rho_fine
        dflux_dt = float(simpson(integrand, x=rho_fine))
        e_phi[i] = -dflux_dt / (2.0 * np.pi * rho_max)
    return e_phi

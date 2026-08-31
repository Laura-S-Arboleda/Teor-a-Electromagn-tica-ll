from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from physics import (
    MagnetParams,
    CoilParams,
    magnet_position,
    magnet_velocity,
    magnetic_flux,
)


@dataclass
class SimulationConfig:
    """Parámetros numéricos de la simulación."""

    t_final: float = 5.0     # Duración total de la simulación [s]
    n_steps: int = 1000       # Número de pasos temporales


@dataclass
class SimulationResult:
    """Resultado completo de una corrida de la simulación."""

    t: np.ndarray
    z_m: np.ndarray
    v_m: np.ndarray
    phi_b: np.ndarray
    dphi_dt: np.ndarray
    emf: np.ndarray
    mag: MagnetParams
    coil: CoilParams
    config: SimulationConfig


def run_simulation(mag: MagnetParams, coil: CoilParams,
                    config: SimulationConfig) -> SimulationResult:
    """Ejecuta la simulación completa y devuelve todas las series temporales.

    Pasos:
      1. Se genera la malla temporal uniforme t = [0, t_final] con n_steps
         puntos (dt = t_final / (n_steps - 1)).
      2. Se calcula z_m(t) y v_m(t) de forma analítica (cinemática impuesta).
      3. Se calcula Phi_B(t) evaluando magnetic_flux(z_m(t_i)) para cada
         instante (integración espacial de Simpson en cada paso temporal).
      4. Se calcula d(Phi_B)/dt mediante diferenciación numérica
         (np.gradient), NO analíticamente, para que el resultado sea
         verificable independientemente del modelo de campo.
      5. La FEM se define como EMF(t) = -d(Phi_B)/dt.

    Efectos numéricos esperables (ver Sección 12 del proyecto / README):
      - Si n_steps es bajo, la derivada numérica introduce error de
        truncamiento notable, especialmente en los picos de Phi_B(t).
      - Los extremos del arreglo (primer y último punto) usan diferencias
        de un solo lado (menor orden de precisión) por construcción de
        np.gradient.
    """
    if config.n_steps < 5:
        raise ValueError("n_steps debe ser >= 5 para una derivada numérica razonable.")
    if config.t_final <= 0:
        raise ValueError("t_final debe ser positivo.")

    t = np.linspace(0.0, config.t_final, config.n_steps)
    z_m = magnet_position(t, mag)
    v_m = magnet_velocity(t, mag)

    phi_b = np.empty_like(t)
    for i, zi in enumerate(z_m):
        phi_b[i] = magnetic_flux(float(zi), mag, coil)

    dt = t[1] - t[0]
    dphi_dt = np.gradient(phi_b, dt)
    emf = -dphi_dt

    return SimulationResult(
        t=t, z_m=z_m, v_m=v_m, phi_b=phi_b, dphi_dt=dphi_dt, emf=emf,
        mag=mag, coil=coil, config=config,
    )


# ---------------------------------------------------------------------------
# Utilidades de validación numérica (Sección 12)
# ---------------------------------------------------------------------------

def convergence_check(mag: MagnetParams, coil: CoilParams,
                       t_final: float, step_counts: list[int]) -> dict:
    """Evalúa la convergencia del valor máximo de |EMF| al refinar dt.

    Devuelve un diccionario {n_steps: max(|EMF|)} que permite verificar que
    el resultado se estabiliza (converge) a medida que aumenta la
    resolución temporal, tal como exige la validación numérica del
    proyecto. No se inventan valores: cada entrada requiere ejecutar
    realmente la simulación con esa resolución.
    """
    results = {}
    for n in step_counts:
        cfg = SimulationConfig(t_final=t_final, n_steps=n)
        res = run_simulation(mag, coil, cfg)
        results[n] = float(np.max(np.abs(res.emf)))
    return results


def lenz_sign_check(result: SimulationResult) -> bool:
    """Verifica que EMF(t) = -dPhi/dt en cada punto (consistencia interna).

    Comprueba, dentro de una tolerancia numérica, que el arreglo `emf`
    almacenado efectivamente satisface la definición de la Ley de Faraday
    respecto al arreglo `dphi_dt` ya calculado. Esto NO es una prueba física
    independiente (ambos provienen del mismo cálculo), sino una verificación
    de integridad de que no se introdujo ningún error de signo o de
    unidades al propagar los resultados por la interfaz gráfica.
    """
    return bool(np.allclose(result.emf, -result.dphi_dt, rtol=1e-9, atol=1e-12))


def quiescent_flux_check(result: SimulationResult, window_frac: float = 0.02) -> float:
    """Devuelve el valor RMS de la FEM en el instante de velocidad mínima.

    Cuando el imán pasa por un extremo de su oscilación (v_m ~ 0), el flujo
    varía lentamente y se espera que |EMF| sea pequeña en esa vecindad.
    Esta función no afirma un resultado: calcula la métrica para que el
    usuario la contraste con la simulación real al ejecutarla.
    """
    idx_min_v = int(np.argmin(np.abs(result.v_m)))
    n = len(result.t)
    half_window = max(1, int(window_frac * n / 2))
    lo = max(0, idx_min_v - half_window)
    hi = min(n, idx_min_v + half_window + 1)
    return float(np.sqrt(np.mean(result.emf[lo:hi] ** 2)))

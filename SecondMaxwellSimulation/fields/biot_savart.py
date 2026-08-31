from __future__ import annotations

import numpy as np
from scipy.constants import mu_0


def biot_savart_field(
    observation_points: np.ndarray,
    source_points: np.ndarray,
    segment_vectors: np.ndarray,
    current: float,
) -> np.ndarray:
    """
    Calcula el campo magnético B por Biot-Savart para una espira discretizada.

    Parameters
    ----------
    observation_points : ndarray, shape (M, 3)
        Puntos donde evaluar el campo.
    source_points : ndarray, shape (N, 3)
        Puntos representativos de los segmentos de corriente.
        Usualmente son los puntos medios.
    segment_vectors : ndarray, shape (N, 3)
        Vectores segmento Δl_i.
    current : float
        Corriente en amperios.

    Returns
    -------
    ndarray, shape (M, 3)
        Campo magnético en cada punto de observación.
    """
    observation_points = np.atleast_2d(np.asarray(observation_points, dtype=float))
    source_points = np.atleast_2d(np.asarray(source_points, dtype=float))
    segment_vectors = np.atleast_2d(np.asarray(segment_vectors, dtype=float))

    if source_points.shape != segment_vectors.shape:
        raise ValueError("source_points y segment_vectors deben tener la misma forma.")

    prefactor = mu_0 * current / (4.0 * np.pi)
    B = np.zeros_like(observation_points, dtype=float)

    for i, r_obs in enumerate(observation_points):
        r_vec = r_obs - source_points
        norms = np.linalg.norm(r_vec, axis=1)

        if np.any(norms == 0.0):
            raise ZeroDivisionError(
                "Un punto de observación coincide con una fuente discreta."
            )

        cross = np.cross(segment_vectors, r_vec)
        contrib = cross / norms[:, None] ** 3
        B[i] = prefactor * np.sum(contrib, axis=0)

    return B


def segment_midpoints(points: np.ndarray) -> np.ndarray:
    """
    Calcula los puntos medios de los segmentos de una poligonal cerrada.
    """
    points = np.asarray(points, dtype=float)
    next_points = np.roll(points, shift=-1, axis=0)
    return 0.5 * (points + next_points)
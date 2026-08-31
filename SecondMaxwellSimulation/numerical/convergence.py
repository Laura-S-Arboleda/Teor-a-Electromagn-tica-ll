from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class ConvergenceStudyResult:
    """
    Resultado de un estudio de convergencia.

    Attributes
    ----------
    num_segments : int
        Número de segmentos de la espira.
    grid_n : int
        Resolución de la malla.
    max_abs : float
        Error máximo absoluto.
    rms : float
        Error RMS.
    mean_abs : float
        Error medio absoluto.
    """
    num_segments: int
    grid_n: int
    max_abs: float
    rms: float
    mean_abs: float


def build_results_table(results: Iterable[ConvergenceStudyResult]) -> list[dict]:
    """
    Convierte una colección de resultados en una lista de diccionarios.
    """
    return [
        {
            "num_segments": r.num_segments,
            "grid_n": r.grid_n,
            "max_abs": r.max_abs,
            "rms": r.rms,
            "mean_abs": r.mean_abs,
        }
        for r in results
    ]


def summarize_errors(values: np.ndarray) -> dict:
    """
    Resume estadísticas básicas de un conjunto de errores.
    """
    values = np.asarray(values, dtype=float)
    abs_values = np.abs(values)

    return {
        "max_abs": float(np.max(abs_values)),
        "mean_abs": float(np.mean(abs_values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "median_abs": float(np.median(abs_values)),
        "std": float(np.std(values)),
    }
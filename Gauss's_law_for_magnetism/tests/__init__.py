from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoopConfig:
    """
    Configuración base de la espira circular.
    """
    radius: float = 1.0
    current: float = 1.0
    num_segments: int = 200
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class GridConfig:
    """
    Configuración de la malla espacial.
    """
    extent: float = 1.5
    n: int = 17


@dataclass(frozen=True)
class VisualizationConfig:
    """
    Configuración visual.
    """
    figsize_2d: tuple[int, int] = (8, 6)
    figsize_3d: tuple[int, int] = (10, 8)
    cmap: str = "viridis"
    field_density: float = 1.2
    animation_interval_ms: int = 40
    animation_frames: int = 120


@dataclass(frozen=True)
class SimulationConfig:
    """
    Configuración general de simulación.
    """
    loop: LoopConfig = field(default_factory=LoopConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    vis: VisualizationConfig = field(default_factory=VisualizationConfig)
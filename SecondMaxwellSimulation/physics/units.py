from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SIUnits:
    """
    Convenciones básicas de unidades SI para el proyecto.

    Este módulo no implementa conversiones complejas;
    solo centraliza el significado físico de las magnitudes.
    """

    length: str = "m"
    time: str = "s"
    mass: str = "kg"
    current: str = "A"
    temperature: str = "K"
    charge: str = "C"
    magnetic_field: str = "T"
    electric_field: str = "V/m"
    voltage: str = "V"
    frequency: str = "Hz"
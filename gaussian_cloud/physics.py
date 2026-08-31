import numpy as np

EPS0_NORMALIZED = 1.0            # permitividad del vacío, unidades normalizadas (valor usado por defecto)
EPS0_SI = 8.8541878128e-12       # permitividad del vacío real (F/m), solo como referencia informativa


def gaussian_charge_density(X: np.ndarray, Y: np.ndarray, rho0: float,
                             sigma: float, x0: float, y0: float) -> np.ndarray:
    """
    Calcula una densidad de carga con perfil gaussiano radial en 2D:

        rho(x,y) = rho0 * exp( -((x-x0)^2 + (y-y0)^2) / (2 sigma^2) )

    Parameters
    ----------
    X, Y : np.ndarray
        Mallas de coordenadas (meshgrid, indexing='ij') donde se evalúa la densidad.
    rho0 : float
        Amplitud de la densidad de carga en el centro de la nube. Como este es un
        modelo bidimensional, rho0 tiene unidades de "carga por unidad de área" en
        el sistema de unidades normalizado empleado. rho0 > 0 representa una nube
        de carga positiva; rho0 < 0, una nube de carga negativa.
    sigma : float
        Desviación estándar de la distribución gaussiana; controla la extensión
        espacial de la nube. Un sigma pequeño concentra la carga cerca de (x0, y0);
        un sigma grande la dispersa sobre una región más amplia del dominio.
    x0, y0 : float
        Coordenadas del centro de la nube de carga.

    Returns
    -------
    np.ndarray
        Matriz con la densidad de carga evaluada en cada punto de la malla.

    Notas físicas y limitaciones
    -----------------------------
    Este modelo es una simplificación bidimensional y puede interpretarse de
    dos maneras distintas, que NO son equivalentes:

    (a) Como una distribución de carga verdaderamente 2D (por ejemplo, una
        lámina delgada cargada con densidad superficial variable). En ese
        caso, la ecuación de Poisson 2D resuelta en este proyecto es exacta
        para ese problema físico.

    (b) Como un corte bidimensional (por ejemplo, el plano z=0) de una nube
        de carga 3D con simetría esférica. En este segundo caso, el
        resultado NO reproduce exactamente el campo tridimensional real: el
        operador de Poisson en 2D tiene una solución fundamental logarítmica,
        mientras que en 3D es de tipo 1/r. Por lo tanto, los valores absolutos
        de V y E obtenidos aquí no deben interpretarse como "el campo 3D real
        de una nube esférica", sino como el resultado de un modelo 2D
        autoconsistente que ilustra correctamente el comportamiento
        cualitativo de la Ley de Gauss.

    En ambos casos, sigma controla el tamaño físico efectivo de la nube y
    rho0 su intensidad.
    """
    if sigma <= 0:
        raise ValueError("sigma debe ser estrictamente positivo.")
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    return rho0 * np.exp(-r2 / (2.0 * sigma ** 2))


def total_charge(rho: np.ndarray, dx: float, dy: float) -> float:
    """
    Integra numéricamente la densidad de carga sobre todo el dominio
    mediante una suma de Riemann (regla del rectángulo):

        Q_total ≈ sum(rho) * dx * dy

    Parameters
    ----------
    rho : np.ndarray
        Densidad de carga en cada nodo de la malla.
    dx, dy : float
        Espaciado de la malla en x e y.

    Returns
    -------
    float
        Carga total integrada numéricamente en el dominio simulado.
    """
    return float(np.sum(rho) * dx * dy)

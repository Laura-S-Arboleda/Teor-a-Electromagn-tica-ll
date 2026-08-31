import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def build_laplacian_2d(nx: int, ny: int, dx: float, dy: float) -> sparse.csr_matrix:
    """
    Construye la matriz dispersa del operador laplaciano discreto 2D usando
    un esquema de diferencias finitas de 5 puntos, sobre una malla de nx*ny
    nodos, con condiciones de frontera de Dirichlet (V = 0 en el borde).

    Discretización en un nodo interior (i, j):

        (d^2V/dx^2)_{i,j} ≈ (V[i+1,j] - 2 V[i,j] + V[i-1,j]) / dx^2
        (d^2V/dy^2)_{i,j} ≈ (V[i,j+1] - 2 V[i,j] + V[i,j-1]) / dy^2

        (laplaciano V)_{i,j} ≈ suma de ambos términos.

    Los nodos (i, j) se numeran en un vector 1D mediante k = i*ny + j
    (orden tipo 'C', con i indexando x y j indexando y).

    Las filas correspondientes a nodos de frontera se reemplazan por filas
    de identidad, de modo que el sistema lineal L @ V_vec = b impone
    directamente V = 0 en el borde cuando b también se anula ahí (ver
    `solve_poisson`).

    Parameters
    ----------
    nx, ny : int
        Número de nodos de la malla en x e y.
    dx, dy : float
        Espaciado de la malla en x e y.

    Returns
    -------
    scipy.sparse.csr_matrix
        Matriz laplaciana discreta de tamaño (nx*ny, nx*ny).
    """
    N = nx * ny
    inv_dx2 = 1.0 / dx ** 2
    inv_dy2 = 1.0 / dy ** 2

    diagonals = []
    offsets = []

    # Diagonal principal: -2*(1/dx^2 + 1/dy^2)
    main = -2.0 * (inv_dx2 + inv_dy2) * np.ones(N)
    diagonals.append(main)
    offsets.append(0)

    # Vecinos en y (offset ±1 dentro del vector aplanado).
    # Deben anularse las conexiones "ficticias" que unen el final de una
    # fila i (j = ny-1) con el inicio de la fila i+1 (j = 0), ya que esos
    # nodos no son vecinos físicos en la malla 2D.
    upper_y = inv_dy2 * np.ones(N - 1)
    lower_y = inv_dy2 * np.ones(N - 1)
    for i in range(nx - 1):
        idx = (i + 1) * ny - 1
        if 0 <= idx < N - 1:
            upper_y[idx] = 0.0
            lower_y[idx] = 0.0
    diagonals.append(upper_y)
    offsets.append(1)
    diagonals.append(lower_y)
    offsets.append(-1)

    # Vecinos en x (offset ±ny): conectan (i,j) con (i+1,j); no requieren
    # corrección adicional porque el salto de tamaño ny siempre corresponde
    # a un vecino físico válido dentro del rango construido.
    upper_x = inv_dx2 * np.ones(N - ny)
    lower_x = inv_dx2 * np.ones(N - ny)
    diagonals.append(upper_x)
    offsets.append(ny)
    diagonals.append(lower_x)
    offsets.append(-ny)

    L = sparse.diags(diagonals, offsets, shape=(N, N), format="lil")

    # Imponer Dirichlet V = 0 en el borde del dominio: filas de frontera -> identidad.
    boundary_mask = np.zeros((nx, ny), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_idx = np.where(boundary_mask.ravel())[0]

    for k in boundary_idx:
        L.rows[k] = [k]
        L.data[k] = [1.0]

    return L.tocsr()


def solve_poisson(rho: np.ndarray, dx: float, dy: float, eps0: float) -> np.ndarray:
    """
    Resuelve numéricamente la ecuación de Poisson:

        laplaciano(V) = -rho / eps0

    con condiciones de frontera de Dirichlet V = 0 en el borde del dominio,
    mediante un método directo (factorización dispersa vía `spsolve`).

    Justificación de la condición de frontera
    ------------------------------------------
    Se asume que el dominio simulado es suficientemente grande respecto a la
    extensión de la nube de carga (sigma) para que el potencial en el borde
    sea aproximadamente nulo, ya que el potencial generado por una
    distribución de carga localizada decae con la distancia al alejarse de
    la fuente. Esta aproximación introduce un error de frontera que
    disminuye a medida que el dominio (L) es mucho mayor que sigma, y que
    puede notarse como una ligera distorsión de la Ley de Gauss cerca de los
    bordes del dominio simulado.

    Parameters
    ----------
    rho : np.ndarray, forma (nx, ny)
        Densidad de carga en cada nodo de la malla.
    dx, dy : float
        Espaciado de la malla.
    eps0 : float
        Permitividad del vacío, en las unidades del sistema empleado
        (por defecto, unidades normalizadas con eps0 = 1; ver physics.py).

    Returns
    -------
    np.ndarray, forma (nx, ny)
        Potencial eléctrico V en cada nodo de la malla.
    """
    nx, ny = rho.shape
    L = build_laplacian_2d(nx, ny, dx, dy)

    b = (-rho / eps0).ravel()

    boundary_mask = np.zeros((nx, ny), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    b[boundary_mask.ravel()] = 0.0   # V = 0 impuesto directamente en el borde

    V_flat = spsolve(L, b)
    return V_flat.reshape(nx, ny)


def compute_electric_field(V: np.ndarray, dx: float, dy: float):
    """
    Calcula el campo eléctrico E = -grad(V) mediante diferencias finitas
    centradas de segundo orden (np.gradient), consistente con la convención
    de ejes usada en todo el proyecto: eje 0 del arreglo -> x, eje 1 -> y.

    Parameters
    ----------
    V : np.ndarray, forma (nx, ny)
        Potencial eléctrico.
    dx, dy : float
        Espaciado de la malla.

    Returns
    -------
    Ex, Ey : np.ndarray, forma (nx, ny)
        Componentes cartesianas del campo eléctrico.
    """
    dVdx, dVdy = np.gradient(V, dx, dy, edge_order=2)
    Ex = -dVdx
    Ey = -dVdy
    return Ex, Ey


def compute_divergence(Ex: np.ndarray, Ey: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Calcula la divergencia numérica del campo eléctrico:

        div(E) = dEx/dx + dEy/dy

    usando diferencias finitas centradas de segundo orden.
    """
    dExdx, _ = np.gradient(Ex, dx, dy, edge_order=2)
    _, dEydy = np.gradient(Ey, dx, dy, edge_order=2)
    return dExdx + dEydy


def gauss_law_error(divE: np.ndarray, rho: np.ndarray, eps0: float) -> dict:
    """
    Compara la divergencia numérica del campo eléctrico con rho/eps0, es
    decir, verifica numéricamente la Ley de Gauss local.

    Se reportan dos tipos de error:

    - Error absoluto puntual:  |divE - rho/eps0|  en cada nodo de la malla.
    - Error relativo GLOBAL, definido en norma L2:

          ||divE - rho/eps0||_2 / ||rho/eps0||_2

      Se utiliza una norma global en lugar de un cociente punto a punto para
      evitar divisiones por valores cercanos a cero en las regiones sin
      carga (donde rho/eps0 ≈ 0), que producirían errores relativos locales
      artificialmente enormes sin significado físico real.

    Parameters
    ----------
    divE : np.ndarray
        Divergencia numérica del campo eléctrico.
    rho : np.ndarray
        Densidad de carga.
    eps0 : float
        Permitividad del vacío empleada.

    Returns
    -------
    dict
        'error_map'          : mapa del error absoluto puntual.
        'target'              : mapa de rho/eps0 (valor teórico esperado).
        'max_abs_error'       : máximo error absoluto puntual.
        'rms_abs_error'       : error absoluto cuadrático medio.
        'relative_l2_error'   : error relativo global en norma L2
                                 (NaN si rho/eps0 es idénticamente cero).
    """
    target = rho / eps0
    error_map = np.abs(divE - target)

    max_abs_error = float(np.max(error_map))
    rms_abs_error = float(np.sqrt(np.mean(error_map ** 2)))

    norm_target = np.linalg.norm(target.ravel())
    norm_diff = np.linalg.norm((divE - target).ravel())
    if norm_target > 1e-14:
        relative_l2_error = float(norm_diff / norm_target)
    else:
        relative_l2_error = float("nan")

    return {
        "error_map": error_map,
        "target": target,
        "max_abs_error": max_abs_error,
        "rms_abs_error": rms_abs_error,
        "relative_l2_error": relative_l2_error,
    }

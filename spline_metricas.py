"""
SCRIPT: spline_metricas.py
WORKSHOP 2: ALGORITMOS NUMÉRICOS

Universidad del Quindío 
facultad de Ingeniería 

El módulo proporciona rutinas para la construcción de Splines Cúbicos por tramos bajo condiciones 
de frontera natural, así como un conjunto de funciones de evaluación cuantitativa de error.

OBJETIVOS:
    - Reutilización modular, ya que funciona como un paquete independiente
    sin variables globales, permitiendo su importacion desde cualquier script.
    - Construir polinomios por tramos de grado 3 garantizando suavidad  
    y aplicando la condición de frontera de derivada segunda nula en los extremos.
    - Facilitar la comparación directa entre polinomios globales (Lagrange y Newton)
    y construcciones por tramos (Spline) utilizando exactamente el mismo conjunto de datos reales.

    AUTORES:
    Dana Sarai Soto
    Laura Sofia Restrepo

Version: 1,0
"""

import numpy as np

def construir_spline_cubico_natural(x_nodes, y_nodes):
    """
    Construye los coeficientes del Spline Cúbico Natural C2 (S''(x_0) = 0, S''(x_n) = 0).

    Cada tramo i para x en [x_i, x_{i+1}] se define como:
    S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3

    Parámetros:
    -----------
    x_nodes : array_like (n+1,)
        Nodos de interpolación estrictamente crecientes.
    y_nodes : array_like (n+1,)
        Valores de la función observada en los nodos.

    Retorna:
    --------
    dict
        Diccionario con los coeficientes a, b, c, d por tramo, nodos x, y el sistema
        tridiagonal resuelto para las segundas derivadas.
    """
    x = np.array(x_nodes, dtype=float).flatten()
    y = np.array(y_nodes, dtype=float).flatten()
    n = len(x) - 1

    if len(y) != len(x):
        raise ValueError("Las dimensiones de x_nodes e y_nodes deben coincidir.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Los nodos x_nodes deben estar ordenados en forma estrictamente creciente.")

    h = np.diff(x)  # h_i = x_{i+1} - x_i

    # Construcción del sistema tridiagonal A * c = rhs para c_i = S''(x_i)/2
    A = np.zeros((n + 1, n + 1), dtype=float)
    rhs = np.zeros(n + 1, dtype=float)

    # Condición de Frontera Natural: S''(x_0) = 0 y S''(x_n) = 0 -> c_0 = 0, c_n = 0
    A[0, 0] = 1.0
    A[n, n] = 1.0

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Resolución del sistema tridiagonal
    c = np.linalg.solve(A, rhs)

    # Cálculo de coeficientes a_i, b_i, d_i por tramo
    a = y[:-1].copy()
    b = np.zeros(n, dtype=float)
    d = np.zeros(n, dtype=float)

    for i in range(n):
        b[i] = (y[i + 1] - y[i]) / h[i] - (h[i] / 3.0) * (2.0 * c[i] + c[i + 1])
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

    return {
        'a': a,
        'b': b,
        'c': c[:-1],
        'd': d,
        'c_nodos': c,
        'x_nodes': x,
        'y_nodes': y,
        'n_tramos': n,
        'exito': True
    }


def evaluar_spline(spline_dict, x_eval, permitir_extrapolacion=False):
    """
    Evalúa el Spline Cúbico en un valor o vector x_eval.

    Parámetros:
    -----------
    spline_dict : dict
        Estructura retornada por `construir_spline_cubico_natural`.
    x_eval : float o array_like
        Puntos a evaluar.
    permitir_extrapolacion : bool, opcional (por defecto False)
        Si es False, desactiva la extrapolación fuera de [x_0, x_n] retornando NaN.

    Retorna:
    --------
    float o np.ndarray
        Valores evaluados del spline.
    """
    x_nodes = spline_dict['x_nodes']
    a = spline_dict['a']
    b = spline_dict['b']
    c = spline_dict['c']
    d = spline_dict['d']
    n = spline_dict['n_tramos']

    x_e = np.atleast_1d(x_eval)
    y_eval = np.full_like(x_e, np.nan, dtype=float)

    x_min, x_max = x_nodes[0], x_nodes[-1]

    for idx, xi in enumerate(x_e):
        if not permitir_extrapolacion and (xi < x_min or xi > x_max):
            continue

        # Seleccionar el tramo adecuado
        if xi <= x_min:
            i = 0
        elif xi >= x_max:
            i = n - 1
        else:
            i = np.searchsorted(x_nodes, xi, side='right') - 1
            i = min(i, n - 1)

        dx = xi - x_nodes[i]
        y_eval[idx] = a[i] + b[i] * dx + c[i] * (dx**2) + d[i] * (dx**3)

    return float(y_eval[0]) if np.isscalar(x_eval) else y_eval


def calcular_metricas_error(y_real, y_pred):
    """
    Calcula el error con signo y las métricas de discrepancia sobre observaciones reservadas:
    - Error con signo: e_i = y_pred_i - y_real_i
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - E_max: Error máximo absoluto

    Parámetros:
    -----------
    y_real : array_like (m,)
        Valores reales o de validación observados.
    y_pred : array_like (m,)
        Valores estimaciones por el interpolador.

    Retorna:
    --------
    dict
        Diccionario con 'errores_signo', 'MAE', 'RMSE', 'E_max' y 'm_observaciones'.
    """
    y_r = np.array(y_real, dtype=float).flatten()
    y_p = np.array(y_pred, dtype=float).flatten()

    if len(y_r) != len(y_p):
        raise ValueError("Las dimensiones de y_real y y_pred deben ser idénticas.")

    # Filtro para omitir valores NaN de extrapolaciones desactivadas
    mask = ~np.isnan(y_p)
    if not np.any(mask):
        return {
            'errores_signo': np.array([]),
            'MAE': np.nan,
            'RMSE': np.nan,
            'E_max': np.nan,
            'm_observaciones': 0
        }

    y_r = y_r[mask]
    y_p = y_p[mask]
    m = len(y_r)

    # Error con signo: e_i = \hat{y}_i - y_i
    e_i = y_p - y_r

    mae = float(np.mean(np.abs(e_i)))
    rmse = float(np.sqrt(np.mean(e_i**2)))
    e_max = float(np.max(np.abs(e_i)))

    return {
        'errores_signo': e_i,
        'MAE': mae,
        'RMSE': rmse,
        'E_max': e_max,
        'm_observaciones': m
    }


def reportar_comparacion_metricas(dict_metodos, y_real, unidad_y=""):
    """
    Imprime una tabla comparativa de las métricas MAE, RMSE y E_max para múltiples
    métodos de interpolación sobre el mismo conjunto de validación.

    Parámetros:
    -----------
    dict_metodos : dict
        Diccionario { 'Nombre_Metodo': y_pred_array }.
    y_real : array_like
        Vector de observaciones reservadas reales.
    unidad_y : str
        Unidad física de la variable para el reporte.
    """
    print("=" * 75)
    print("TABLA COMPARATIVA DE MÉTRICAS SOBRE OBSERVACIONES RESERVADAS")
    print("=" * 75)
    print(f"{'Método':<22} | {'MAE (' + unidad_y + ')':>14} | {'RMSE (' + unidad_y + ')':>14} | {'E_máx (' + unidad_y + ')':>14}")
    print("-" * 75)

    for nombre, y_pred in dict_metodos.items():
        met = calcular_metricas_error(y_real, y_pred)
        print(f"{nombre:<22} | {met['MAE']:14.6f} | {met['RMSE']:14.6f} | {met['E_max']:14.6f}")

    print("=" * 75)
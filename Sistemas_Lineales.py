""" 
SCRIPT: Sistemas_Lineales.py
WORKSHOP 2: ALGORITMOS NUMÉRICOS

Universidad del Quindío 
facultad de Ingeniería 

Este módulo proporciona funciones reutilizables para la resolución 
numérica de sistemas de ecuaciones lineales de la forma Ax=b
mediante el metodo de eliminacion Gaussiana con pivote parcial por filas.

OBJETIVOS:
    - Reutilización modular, ya que funciona como un paquete independiente
    sin variables globales, permitiendo su importacion desde cualquier script.
    - Solución de sistemas lineales (n x n) arantizando la estabilidad numérica 
    frente a pivotes nulos o de magnitud pequeña mediante intercambio dinámico de filas.
    - Manejar condiciones de fallo devolviendo códigos de estado legibles y mensajes 
    informativos cuando el sistema sea singular, indeterminado o dimensionalmente incompatible.

AUTORES:
    Dana Sarai Soto
    Laura Sofia Restrepo

Version: 1,0
"""


import numpy as np

def eliminacion_gaussiana(A, b, tol=1e-12):
    """
    Resuelve un sistema de ecuaciones lineales Ax = b mediante Eliminación Gaussiana
    con pivoteo parcial por filas y calcula el vector residual r = Ax - b.

    Parámetros:
    -----------
    A : array_like (n, n)
        Matriz de coeficientes del sistema.
    b : array_like (n,) o (n, 1)
        Vector de términos independientes.
    tol : float, opcional (por defecto 1e-12)
        Tolerancia numérica para identificar pivotes nulos o matrices singulares.

    Retorna:
    --------
    dict
        Diccionario con la estructura de resultados y condiciones de salida:
        - 'solucion'        : np.ndarray con el vector solución x (None si falla).
        - 'residual'        : np.ndarray con el vector residual r = A*x - b.
        - 'norma_residual'  : float con la norma euclidiana ||r||_2.
        - 'exito'           : bool, True si la ejecución finalizó correctamente.
        - 'mensaje'         : str con el estado de convergencia o causa de fallo.
        - 'pasos'           : list de dicts con el historial de pivotes, intercambios
                              de fila y matrices aumentadas intermedias.
    """
    A_mat = np.array(A, dtype=float)
    b_vec = np.array(b, dtype=float).flatten()
    n = len(b_vec)

    # Validación dimensional
    if A_mat.shape != (n, n):
        return {
            'solucion': None,
            'residual': None,
            'norma_residual': None,
            'exito': False,
            'mensaje': f"Dimensiones incompatibles: A es {A_mat.shape} y b tiene longitud {n}.",
            'pasos': []
        }

    # Construcción de la matriz aumentada [A | b]
    Ab = np.hstack([A_mat, b_vec.reshape(-1, 1)])
    pasos = []

    # 1. Eliminación Hacia Adelante con Pivoteo Parcial
    for k in range(n - 1):
        # Selección de pivote con mayor magnitud absoluta en la columna k
        max_idx = np.argmax(np.abs(Ab[k:, k])) + k
        val_pivote = Ab[max_idx, k]

        # Comprobación de matriz singular
        if np.abs(val_pivote) < tol:
            return {
                'solucion': None,
                'residual': None,
                'norma_residual': None,
                'exito': False,
                'mensaje': f"Fallo por matriz singular o mal condicionada: pivote nulo en columna {k}.",
                'pasos': pasos
            }

        # Intercambio de filas si el pivote máximo no está en la diagonal
        hubo_intercambio = False
        if max_idx != k:
            Ab[[k, max_idx]] = Ab[[max_idx, k]]
            hubo_intercambio = True

        # Eliminación de elementos por debajo de la diagonal
        multiplicadores = []
        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            multiplicadores.append(factor)
            Ab[i, k:] -= factor * Ab[k, k:]

        pasos.append({
            'etapa': k + 1,
            'fila_pivote': k,
            'fila_intercambiada': max_idx if hubo_intercambio else None,
            'pivote_elegido': float(Ab[k, k]),
            'multiplicadores': multiplicadores,
            'matriz_aumentada': Ab.copy()
        })

    # Verificación del último pivote en la diagonal
    if np.abs(Ab[n - 1, n - 1]) < tol:
        return {
            'solucion': None,
            'residual': None,
            'norma_residual': None,
            'exito': False,
            'mensaje': "Fallo por matriz singular: el último elemento diagonal es nulo.",
            'pasos': pasos
        }

    # 2. Sustitución Regresiva
    x = np.zeros(n, dtype=float)
    x[n - 1] = Ab[n - 1, n] / Ab[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        suma = np.dot(Ab[i, i + 1:n], x[i + 1:n])
        x[i] = (Ab[i, n] - suma) / Ab[i, i]

    # 3. Comprobación y cálculo del vector residual r = Ax - b
    residual = np.dot(A_mat, x) - b_vec
    norma_residual = float(np.linalg.norm(residual, ord=2))

    return {
        'solucion': x,
        'residual': residual,
        'norma_residual': norma_residual,
        'exito': True,
        'mensaje': "Solución obtenida con éxito mediante Eliminación Gaussiana con pivoteo parcial.",
        'pasos': pasos
    }


def reportar_residual(resultado, unidades_b=None):
    """
    Imprime la interpretación del vector residual Ax - b y la norma de error.

    Parámetros:
    -----------
    resultado : dict
        Salida retornada por la función `eliminacion_gaussiana`.
    unidades_b : list o str, opcional
        Unidades físicas de las ecuaciones del sistema.
    """
    if not resultado['exito']:
        print(f"No se puede calcular el residual: {resultado['mensaje']}")
        return

    res = resultado['residual']
    norma = resultado['norma_residual']

    print("=" * 60)
    print("INSPECCIÓN DEL VECTOR RESIDUAL r = A*x - b")
    print("=" * 60)
    for i, r_i in enumerate(res):
        u = unidades_b[i] if isinstance(unidades_b, list) else (unidades_b or "")
        print(f"Ecuación {i+1}: r[{i+1}] = {r_i:+.6e} {u}")
    print(f"\nNorma Euclidiana ||r||_2: {norma:.6e}")
    print("=" * 60)
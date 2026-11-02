"""
SCRIPT: Interpolacion_Lagrange.py
WORKSHOP 2: ALGORITMOS NUMÉRICOS

Universidad del Quindío 
facultad de Ingeniería 

Este módulo proporciona funciones reutilizables para construir y evaluar polinomios 
interpolantes de Lagrange P_n(x) de cualquier grado n-1, adaptándose dinámicamente a 
conjuntos de 2, 3 o 4 nodos.
El módulo calcula explícitamente los polinomios base de Lagrange L_i(x) y verifica 
que las ponderaciones satisfagan la propiedad cardinal o delta de Kronecker 
(L_i(x_j) = delta_{ij}), garantizando que el polinomio pase exactamente por todos los
puntos de construcción antes de estimar consultas interiores.

OBJETIVOS:
    - Reutilización modular, ya que funciona como un paquete independiente
    sin variables globales, permitiendo su importacion desde cualquier script.
    - Calcular las bases de Lagrange L_i(x) de forma explícita, permitiendo 
    inspeccionar las ponderaciones individuales asociadas a cada nodo en el punto de consulta.
    - Verificar numéricamente la reproducción de nodos.

AUTORES:
    Dana Sarai Soto
    Laura Sofia Restrepo

Version: 1,0
"""

import numpy as np

def bases_lagrange(x_nodes, x_eval):
    """
    Calcula los polinomios base de Lagrange L_i(x) evaluados en un punto o vector x_eval.

    Fórmula:
    L_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}

    Parámetros:
    -----------
    x_nodes : array_like
        Vector con las coordenadas x de los nodos de interpolación.
    x_eval : float o array_like
        Punto o array de puntos donde se evaluarán las bases.

    Retorna:
    --------
    np.ndarray
        Array con las bases L_i evaluadas. Si x_eval es escalar, retorna vector 1D
        de longitud n. Si es vector de m puntos, retorna matriz (m, n).
    """
    x_nodes = np.array(x_nodes, dtype=float).flatten()
    n = len(x_nodes)
    
    # Verificación de nodos x duplicados
    if len(np.unique(x_nodes)) != n:
        raise ValueError("Error: Existen nodos de interpolación x duplicados.")

    x_e = np.atleast_1d(x_eval)
    m = len(x_e)
    L = np.ones((m, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i != j:
                L[:, i] *= (x_e - x_nodes[j]) / (x_nodes[i] - x_nodes[j])

    return L[0] if np.isscalar(x_eval) else L


def evaluar_lagrange(x_nodes, y_nodes, x_eval):
    """
    Evalúa el polinomio interpolante de Lagrange P_n(x) = \sum y_i * L_i(x),
    verificando la validez de los datos y la reproducción de nodos.

    Parámetros:
    -----------
    x_nodes : array_like
        Nodos de construcción (coordenadas x).
    y_nodes : array_like
        Valores observados (coordenadas y).
    x_eval : float o array_like
        Punto o puntos donde se evaluará el polinomio.

    Retorna:
    --------
    dict
        Diccionario estructurado con:
        - 'y_eval'              : float o np.ndarray con P_n(x_eval).
        - 'bases_eval'          : np.ndarray con las ponderaciones L_i(x_eval).
        - 'grado'               : int (n - 1).
        - 'reproduccion_nodos'  : bool, True si P_n(x_i) == y_i y L_i(x_j) == δ_ij.
        - 'matriz_bases_nodos'  : np.ndarray (n, n) evaluando L_i(x_j).
        - 'exito'               : bool.
        - 'mensaje'             : str explicativo.
    """
    x_nodes = np.array(x_nodes, dtype=float).flatten()
    y_nodes = np.array(y_nodes, dtype=float).flatten()

    n = len(x_nodes)

    # Validación de dimensiones
    if len(y_nodes) != n:
        return {
            'y_eval': None,
            'bases_eval': None,
            'grado': None,
            'reproduccion_nodos': False,
            'matriz_bases_nodos': None,
            'exito': False,
            'mensaje': f"Dimensiones incompatibles: {n} nodos en x y {len(y_nodes)} en y."
        }

    if len(np.unique(x_nodes)) != n:
        return {
            'y_eval': None,
            'bases_eval': None,
            'grado': None,
            'reproduccion_nodos': False,
            'matriz_bases_nodos': None,
            'exito': False,
            'mensaje': "Fallo: Se detectaron nodos x duplicados (división por cero)."
        }

    grado = n - 1

    # 1. Evaluación de las bases L_i en x_eval
    bases_eval = bases_lagrange(x_nodes, x_eval)

    # 2. Evaluación de P_n(x_eval) = \sum y_i * L_i(x_eval)
    if np.isscalar(x_eval):
        y_eval = float(np.sum(y_nodes * bases_eval))
    else:
        y_eval = np.dot(bases_eval, y_nodes)

    # 3. Comprobación de reproducción de nodos: L_i(x_j) == δ_{ij} (Matriz Identidad)
    matriz_bases_nodos = bases_lagrange(x_nodes, x_nodes)
    identidad = np.eye(n)
    error_bases = float(np.max(np.abs(matriz_bases_nodos - identidad)))

    # Verificación del valor P_n(x_i) frente a y_i
    y_nodos_reconstruidos = np.dot(matriz_bases_nodos, y_nodes)
    error_reconstruccion = float(np.max(np.abs(y_nodos_reconstruidos - y_nodes)))

    reproduccion_correcta = bool(error_bases < 1e-10 and error_reconstruccion < 1e-10)

    mensaje = f"Interpolación de Lagrange de grado {grado} completada con éxito."
    if not reproduccion_correcta:
        mensaje += " Advertencia: La reproducción de nodos registró discrepancias numéricas."

    return {
        'y_eval': y_eval,
        'bases_eval': bases_eval,
        'grado': grado,
        'reproduccion_nodos': reproduccion_correcta,
        'matriz_bases_nodos': matriz_bases_nodos,
        'exito': True,
        'mensaje': mensaje
    }


def reportar_lagrange(resultado, x_nodes, y_nodes, x_eval, unidad_x="", unidad_y=""):
    """
    Imprime en pantalla el reporte detallado del desarrollo de Lagrange,
    mostrando las bases ponderadas, el valor interpolado y la comprobación de nodos.
    """
    if not resultado['exito']:
        print(f"Error en interpolación: {resultado['mensaje']}")
        return

    print("=" * 75)
    print(f"INTERPOLACIÓN DE LAGRANGE (Polinomio de Grado {resultado['grado']})")
    print("=" * 75)
    print(f"Nodos de construcción ({len(x_nodes)} puntos):")
    for i, (xi, yi) in enumerate(zip(x_nodes, y_nodes)):
        print(f"  Nodo {i}: x_{i} = {xi:10.4f} {unidad_x} | y_{i} = {yi:10.4f} {unidad_y}")
    
    print("\n" + "-" * 75)
    print(f"Evaluación en x = {x_eval} {unidad_x}:")
    
    if np.isscalar(x_eval):
        bases = resultado['bases_eval']
        for i, L_i in enumerate(bases):
            print(f"  Base L_{i}({x_eval}) = {L_i:+.8f}")
        print(f"\n-> Valor Interpolado P_{resultado['grado']}({x_eval}) = {resultado['y_eval']:.8f} {unidad_y}")
    else:
        print(f"  Evaluado sobre un conjunto de {len(x_eval)} puntos.")

    print("\n" + "-" * 75)
    print("VERIFICACIÓN DE REPRODUCCIÓN DE NODOS (L_i(x_j) = δ_ij):")
    mat_id = resultado['matriz_bases_nodos']
    print("Matriz de evaluación de bases en los nodos [L_i(x_j)]:")
    for fila in mat_id:
        print("  [" + "  ".join([f"{val:8.5f}" for val in fila]) + "]")
    
    estado_repr = "CORRECTA (Satisface la propiedad delta de Kronecker)" if resultado['reproduccion_nodos'] else "INCORRECTA"
    print(f"Estado de comprobación: {estado_repr}")
    print("=" * 75)
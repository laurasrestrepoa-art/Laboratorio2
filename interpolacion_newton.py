"""
SCRIPT: interpolacion_newton.py
WORKSHOP 2: ALGORITMOS NUMÉRICOS

Universidad del Quindío 
facultad de Ingeniería 

Ofrece rutinas reutilizables en Python para resolver problemas de interpolación 
de diferencias divididas de Newton. El módulo destaca por su capacidad de construir 
polinomios interpolantes de forma secuencial. Permite evaluar la contribución de 
los términos de orden superior y ofrece una función específica para incorporar un 
nuevo nodo a una estructura preexistente sin recalcular las diferencias divididas 
anteriores, verificando la propiedad fundamental de conservación de coeficientes.

OBJETIVOS:
    - Reutilización modular, ya que funciona como un paquete independiente
    sin variables globales, permitiendo su importacion desde cualquier script.
    - Generar la Tabla de Diferencias Divididas para cualquier conjunto de n puntos medidos.
    - Demostrar la propiedad incremental del método de Newton, agregando nuevos nodos mediante 
    la función (incorporar_nodo_dd)
    - Verificar la reproducción de nodos medidos.
"""

import numpy as np

def construir_tabla_dd(x_nodes, y_nodes):
    """
    Construye la tabla completa de diferencias divididas de Newton para n nodos.

    Parámetros:
    -----------
    x_nodes : array_like
        Vector de coordenadas x de los nodos.
    y_nodes : array_like
        Vector de observaciones y = f(x).

    Retorna:
    --------
    tabla : np.ndarray (n, n)
        Matriz con la tabla de diferencias divididas. La entrada tabla[i, j]
        representa la diferencia dividida de orden j que inicia en el nodo i.
    coeficientes : np.ndarray (n,)
        Vector con los coeficientes a_k = tabla[0, k] del polinomio de Newton.
    """
    x = np.array(x_nodes, dtype=float).flatten()
    y = np.array(y_nodes, dtype=float).flatten()
    n = len(x)

    if len(y) != n:
        raise ValueError(f"Dimensiones incompatibles: {n} nodos en x y {len(y)} en y.")
    if len(np.unique(x)) != n:
        raise ValueError("Error: Existen nodos x duplicados (división por cero).")

    tabla = np.zeros((n, n), dtype=float)
    tabla[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            tabla[i, j] = (tabla[i + 1, j - 1] - tabla[i, j - 1]) / (x[i + j] - x[i])

    coeficientes = tabla[0, :].copy()
    return tabla, coeficientes


def evaluar_newton(x_nodes, coeficientes, x_eval):
    """
    Evalúa el polinomio interpolante de Newton P_n(x) usando la forma anidada de Horner.

    P_n(x) = a_0 + (x - x_0)(a_1 + (x - x_1)(a_2 + ... + (x - x_{n-1})a_n)...)

    Parámetros:
    -----------
    x_nodes : array_like
        Coordenadas x de los nodos utilizados para construir el polinomio.
    coeficientes : array_like
        Coeficientes a_k de las diferencias divididas [f[x0], f[x0,x1], ...].
    x_eval : float o array_like
        Punto o conjunto de puntos a evaluar.

    Retorna:
    --------
    float o np.ndarray
        Valor(es) interpolado(s) P_n(x_eval).
    """
    x = np.array(x_nodes, dtype=float).flatten()
    a = np.array(coeficientes, dtype=float).flatten()
    n = len(a)

    x_e = np.atleast_1d(x_eval)
    y_eval = np.full_like(x_e, a[n - 1], dtype=float)

    for k in range(n - 2, -1, -1):
        y_eval = a[k] + (x_e - x[k]) * y_eval

    return float(y_eval[0]) if np.isscalar(x_eval) else y_eval


def incorporar_nodo_dd(x_nodes_prev, tabla_dd_prev, nuevo_x, nuevo_y):
    """
    Incorpora un nuevo nodo (x_{n+1}, y_{n+1}) a una tabla de diferencias divididas previa
    sin reconstruirla desde cero. Calcula únicamente las diferencias necesarias.

    Parámetros:
    -----------
    x_nodes_prev : array_like
        Nodos x del conjunto anterior (n elementos).
    tabla_dd_prev : np.ndarray (n, n)
        Tabla de diferencias divididas del conjunto anterior.
    nuevo_x : float
        Coordenada x del nuevo nodo a incorporar.
    nuevo_y : float
        Coordenada y del nuevo nodo a incorporar.

    Retorna:
    --------
    x_nodes_actualizados : np.ndarray (n+1,)
        Conjunto ampliado de nodos.
    tabla_actualizada : np.ndarray (n+1, n+1)
        Nueva tabla de diferencias divididas con la fila/columna añadida.
    coeficientes_actualizados : np.ndarray (n+1,)
        Nuevos coeficientes del polinomio de Newton.
    coeficientes_conservados : bool
        True si los primeros n coeficientes coinciden exactamente con los previos.
    """
    x_prev = np.array(x_nodes_prev, dtype=float).flatten()
    n = len(x_prev)

    if nuevo_x in x_prev:
        raise ValueError(f"El nodo x = {nuevo_x} ya existe en el conjunto anterior.")

    x_actualizados = np.append(x_prev, nuevo_x)
    tabla_actualizada = np.zeros((n + 1, n + 1), dtype=float)
    
    # Copiar la tabla anterior en el bloque superior izquierdo
    tabla_actualizada[:n, :n] = tabla_dd_prev.copy()
    
    # Asignar la observación de orden 0 en la posición nueva
    tabla_actualizada[n, 0] = nuevo_y

    # Calcular solo las diferencias divididas afectadas por la incorporación del nuevo nodo
    for j in range(1, n + 1):
        i = n - j
        tabla_actualizada[i, j] = (
            (tabla_actualizada[i + 1, j - 1] - tabla_actualizada[i, j - 1]) / 
            (x_actualizados[i + j] - x_actualizados[i])
        )

    coef_anteriores = tabla_dd_prev[0, :].copy()
    coef_actualizados = tabla_actualizada[0, :].copy()

    # Verificar conservación de coeficientes anteriores: a_0, a_1, ..., a_{n-1}
    coeficientes_conservados = np.allclose(coef_anteriores, coef_actualizados[:n], atol=1e-12)

    return x_actualizados, tabla_actualizada, coef_actualizados, coeficientes_conservados


def evaluar_interpolacion_completa(x_nodes, y_nodes, x_eval):
    """
    Función envolvente que ejecuta el procedimiento de Newton, evalúa en x_eval
    y comprueba la reproducción exacta de nodos.
    """
    x = np.array(x_nodes, dtype=float).flatten()
    y = np.array(y_nodes, dtype=float).flatten()

    tabla, coefs = construir_tabla_dd(x, y)
    y_eval = evaluar_newton(x, coefs, x_eval)

    # Verificación de reproducción de nodos P_n(x_i) == y_i
    y_reconstruidos = evaluar_newton(x, coefs, x)
    error_reconstruccion = float(np.max(np.abs(y_reconstruidos - y)))
    reproduccion_ok = bool(error_reconstruccion < 1e-10)

    return {
        'tabla_dd': tabla,
        'coeficientes': coefs,
        'y_eval': y_eval,
        'grado': len(x) - 1,
        'reproduccion_nodos': reproduccion_ok,
        'exito': True,
        'mensaje': f"Interpolación de Newton de grado {len(x)-1} completada."
    }


def reportar_newton(resultado, x_nodes, y_nodes, x_eval, unidad_x="", unidad_y=""):
    """
    Imprime en pantalla la tabla de diferencias divididas, los coeficientes
    y la evaluación del polinomio de Newton.
    """
    if not resultado['exito']:
        print(f"Error: {resultado['mensaje']}")
        return

    tabla = resultado['tabla_dd']
    coefs = resultado['coeficientes']
    n = len(x_nodes)

    print("=" * 80)
    print(f"INTERPOLACIÓN DE NEWTON (Polinomio de Grado {resultado['grado']})")
    print("=" * 80)
    
    # Imprimir Tabla de Diferencias Divididas
    print("TABLA DE DIFERENCIAS DIVIDIDAS:")
    encabezados = [f"x ({unidad_x})", f"f(x) ({unidad_y})"] + [f"Order {j} DD" for j in range(1, n)]
    header_str = " | ".join([f"{h:>12}" for h in encabezados])
    print(header_str)
    print("-" * len(header_str))

    for i in range(n):
        fila_str = f"{x_nodes[i]:12.4f} | "
        valores_j = []
        for j in range(n - i):
            valores_j.append(f"{tabla[i, j]:12.6f}")
        fila_str += " | ".join(valores_j)
        print(fila_str)

    print("\n" + "-" * 80)
    print("COEFICIENTES DEL POLINOMIO DE NEWTON [a_0, a_1, ..., a_n]:")
    for k, ak in enumerate(coefs):
        print(f"  a_{k} = {ak:+.8e}")

    print("\n" + "-" * 80)
    print(f"EVALUACIÓN EN x = {x_eval} {unidad_x}:")
    print(f"-> Valor Estimado P_{resultado['grado']}({x_eval}) = {resultado['y_eval']:.8f} {unidad_y}")
    
    estado_repr = "CORRECTA (P_n(x_i) = y_i)" if resultado['reproduccion_nodos'] else "INCORRECTA"
    print(f"Reproducción de Nodos: {estado_repr}")
    print("=" * 80)
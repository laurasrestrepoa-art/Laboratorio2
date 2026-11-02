"""
SCRIPT: Sistemas_No_Lineales.py
WORKSHOP 2: ALGORITMOS NUMÉRICOS

Universidad del Quindío 
facultad de Ingeniería 

Este módulo proporciona rutinas computacionales en Python para resolver sistemas de 
ecuaciones no lineales mediante los métodos de Punto Fijo (con actualización simultánea) 
y Newton-Raphson para sistemas.

OBJETIVOS:
    - Resolver sistemas no lineales multivariables garantizando el cumplimiento simultáneo 
    de tolerancias en el cambio de variables y en el residual de las ecuaciones.
    - Reutilización modular, ya que funciona como un paquete independiente
    sin variables globales, permitiendo su importacion desde cualquier script.

AUTORES:
    Dana Sarai Soto
    Laura Sofia Restrepo

Version: 1,0
"""

import numpy as np

def punto_fijo_sistemas(g_func, x0, F_func=None, tol=1e-4, max_iter=100):
    """
    Resuelve un sistema no lineal mediante Punto Fijo con actualización simultánea:
    x^{(k+1)} = g(x^{(k)}).

    Parámetros:
    -----------
    g_func : función
        Función g(x) que recibe un array x y retorna el array g(x) despejado.
    x0 : array_like
        Vector de estimación inicial.
    F_func : función, opcional
        Función F(x) que evalúa el sistema original F(x) = 0 para calcular el residual.
    tol : float, opcional (defecto 1e-4)
        Tolerancia máxima para el cambio entre iteraciones y el residual.
    max_iter : int, opcional (defecto 100)
        Número máximo de iteraciones permitidas.

    Retorna:
    --------
    dict
        Diccionario con el resultado de la convergencia, historial de iteraciones,
        vector de cambio Δx y residual ||F(x)||_inf.
    """
    x_k = np.array(x0, dtype=float).flatten()
    historial = []
    
    exito = False
    mensaje = ""

    for k in range(max_iter):
        x_next = np.array(g_func(x_k), dtype=float).flatten()
        
        # Cambio entre iteraciones ||x^{(k+1)} - x^{(k)}||_inf
        delta_x = x_next - x_k
        norma_cambio = float(np.max(np.abs(delta_x)))
        
        # Evaluación del residual si se provee F(x)
        if F_func is not None:
            residual_vec = np.array(F_func(x_next), dtype=float).flatten()
            norma_residual = float(np.max(np.abs(residual_vec)))
        else:
            residual_vec = None
            norma_residual = float('inf')

        historial.append({
            'iteracion': k + 1,
            'x_k': x_k.copy(),
            'x_next': x_next.copy(),
            'delta_x': delta_x.copy(),
            'norma_cambio': norma_cambio,
            'residual_vec': residual_vec,
            'norma_residual': norma_residual
        })

        # Criterio de parada: cambio y residual <= tol
        criterio_cumplido = (norma_cambio <= tol) and (F_func is None or norma_residual <= tol)
        if criterio_cumplido:
            exito = True
            mensaje = f"Convergencia alcanzada en {k + 1} iteraciones (Tol <= {tol})."
            x_k = x_next
            break

        # Verificación de divergencia / overflow
        if np.isnan(norma_cambio) or np.isinf(norma_cambio) or norma_cambio > 1e10:
            mensaje = f"Divergencia detectada en la iteración {k + 1}: valores desbordados."
            return {
                'solucion': None,
                'exito': False,
                'iteraciones': k + 1,
                'mensaje': mensaje,
                'historial': historial
            }

        x_k = x_next

    if not exito and mensaje == "":
        mensaje = f"Se alcanzó el límite máximo de {max_iter} iteraciones sin satisfacer la tolerancia."

    return {
        'solucion': x_k,
        'exito': exito,
        'iteraciones': len(historial),
        'mensaje': mensaje,
        'historial': historial
    }


def newton_raphson_sistemas(F_func, J_func, x0, tol=1e-4, max_iter=100):
    """
    Resuelve un sistema no lineal F(x) = 0 mediante el Método de Newton-Raphson.
    Resuelve el sistema lineal de correcciones J(x^{(k)}) * Δx = -F(x^{(k)}) 
    sin invertir la matriz Jacobiana explícitamente.

    Parámetros:
    -----------
    F_func : función
        Retorna el vector residual F(x).
    J_func : función
        Retorna la matriz Jacobiana J(x).
    x0 : array_like
        Vector con los valores iniciales de las variables.
    tol : float, opcional (defecto 1e-4)
        Tolerancia conjunta para norma de cambio y residual.
    max_iter : int, opcional (defecto 100)
        Límite de iteraciones.

    Retorna:
    --------
    dict
        Resultados de solución, delta_x, residual, historial y condición de salida.
    """
    x_k = np.array(x0, dtype=float).flatten()
    historial = []
    
    exito = False
    mensaje = ""

    for k in range(max_iter):
        F_k = np.array(F_func(x_k), dtype=float).flatten()
        J_k = np.array(J_func(x_k), dtype=float)

        # Resolver sistema de corrección: J_k * delta_x = -F_k
        try:
            delta_x = np.linalg.solve(J_k, -F_k)
        except np.linalg.LinAlgError:
            return {
                'solucion': None,
                'exito': False,
                'iteraciones': k,
                'mensaje': f"Fallo en iteración {k+1}: Matriz Jacobiana singular o mal condicionada.",
                'historial': historial
            }

        # Actualización de la aproximación: x^{(k+1)} = x^{(k)} + Δx
        x_next = x_k + delta_x
        
        # Evaluación del residual en la nueva aproximación F(x^{(k+1)})
        F_next = np.array(F_func(x_next), dtype=float).flatten()
        
        norma_cambio = float(np.max(np.abs(delta_x)))
        norma_residual = float(np.max(np.abs(F_next)))

        historial.append({
            'iteracion': k + 1,
            'x_k': x_k.copy(),
            'delta_x': delta_x.copy(),
            'x_next': x_next.copy(),
            'norma_cambio': norma_cambio,
            'F_next': F_next.copy(),
            'norma_residual': norma_residual
        })

        # Criterio de parada especificado en la guía (ambos menores o iguales a la tolerancia)
        if norma_cambio <= tol and norma_residual <= tol:
            exito = True
            mensaje = f"Convergencia de Newton-Raphson alcanzada en {k + 1} iteraciones."
            x_k = x_next
            break

        x_k = x_next

    if not exito and mensaje == "":
        mensaje = f"Se alcanzó el límite máximo de {max_iter} iteraciones sin convergencia."

    return {
        'solucion': x_k,
        'exito': exito,
        'iteraciones': len(historial),
        'mensaje': mensaje,
        'historial': historial
    }


def reportar_tabla_iteraciones(resultado, nombres_var=None):
    """
    Imprime en formato tabular el historial de iteraciones, cambios Δx y residuales.
    """
    print("=" * 80)
    print(f"ESTADO DE SALIDA: {resultado['mensaje']}")
    print("=" * 80)
    
    if not resultado['historial']:
        return

    n_vars = len(resultado['historial'][0]['x_k'])
    headers_var = nombres_var if nombres_var else [f"x{i+1}" for i in range(n_vars)]
    
    print(f"{'k':<4} | " + " | ".join([f"{h:>10}" for h in headers_var]) + f" | {'||Δx||_inf':>12} | {'||F(x)||_inf':>12}")
    print("-" * 80)

    for item in resultado['historial']:
        vals_x = " | ".join([f"{val:10.6f}" for val in item['x_next']])
        nc = f"{item['norma_cambio']:12.6e}"
        nr = f"{item['norma_residual']:12.6e}"
        print(f"{item['iteracion']:<4} | {vals_x} | {nc} | {nr}")
    print("=" * 80)
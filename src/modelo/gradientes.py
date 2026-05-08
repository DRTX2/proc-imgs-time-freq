"""Acentuado espacial y operadores de gradiente."""

import math

import numpy as np

from .base import convertir_a_grises
from .utils import matriz_absoluta_manual, normalizar_matriz_uint8, diferencia_absoluta_manual


# Filtros de gradiente y pasa-altas recorridos pixel a pixel.

def _gradiente_roberts_puro(imagen):
    """
    Devuelve solo la magnitud Roberts, sin sumar la imagen original.
    Ejemplo corto: compara diagonales vecinas; si cambian mucho, marca borde.
    """
    alto, ancho = imagen.shape
    resultado = np.zeros((alto, ancho), dtype=np.uint8)
    for fila in range(alto - 1):
        for columna in range(ancho - 1):
            gx = int(imagen[fila, columna]) - int(imagen[fila + 1, columna + 1])
            gy = int(imagen[fila, columna + 1]) - int(imagen[fila + 1, columna])
            gradiente = int(math.sqrt(gx * gx + gy * gy))
            resultado[fila, columna] = min(gradiente, 255)
    return resultado


def _gradiente_prewitt_puro(imagen):
    """
    Devuelve solo la magnitud Prewitt, sin sumar la imagen original.
    Ejemplo corto: Gx compara izquierda/derecha y Gy compara arriba/abajo.
    """
    alto, ancho = imagen.shape
    resultado = np.zeros((alto, ancho), dtype=np.uint8)
    Gx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    Gy = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            suma_gx = 0.0
            suma_gy = 0.0
            for mf in range(3):
                for mc in range(3):
                    pixel = int(imagen[fila - 1 + mf, columna - 1 + mc])
                    suma_gx += pixel * Gx[mf][mc]
                    suma_gy += pixel * Gy[mf][mc]
            gradiente = int(math.sqrt(suma_gx * suma_gx + suma_gy * suma_gy))
            resultado[fila, columna] = min(gradiente, 255)
    return resultado


def _gradiente_sobel_puro(imagen):
    """
    Devuelve solo la magnitud Sobel, sin sumar la imagen original.
    Ejemplo corto: funciona como Prewitt, pero da más peso al centro de la ventana.
    """
    alto, ancho = imagen.shape
    resultado = np.zeros((alto, ancho), dtype=np.uint8)
    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            suma_gx = 0.0
            suma_gy = 0.0
            for mf in range(3):
                for mc in range(3):
                    pixel = int(imagen[fila - 1 + mf, columna - 1 + mc])
                    suma_gx += pixel * Gx[mf][mc]
                    suma_gy += pixel * Gy[mf][mc]
            gradiente = int(math.sqrt(suma_gx * suma_gx + suma_gy * suma_gy))
            resultado[fila, columna] = min(gradiente, 255)
    return resultado


def _gradiente_kirsch_puro(imagen):
    """
    Devuelve la respuesta máxima de Kirsch usando cuatro direcciones.
    Ejemplo corto: si un borde aparece fuerte al norte pero débil al este,
    se conserva la respuesta norte.
    """
    alto, ancho = imagen.shape
    resultado = np.zeros((alto, ancho), dtype=np.uint8)
    kernels = _kernels_kirsch()

    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            maximo = 0.0
            for kernel in kernels:
                respuesta = 0.0
                for mf in range(3):
                    for mc in range(3):
                        pixel = int(imagen[fila - 1 + mf, columna - 1 + mc])
                        respuesta += pixel * kernel[mf][mc]
                respuesta = abs(respuesta)
                if respuesta > maximo:
                    maximo = respuesta
            resultado[fila, columna] = min(int(maximo), 255)
    return resultado


def _laplaciano_puro(imagen):
    """
    Devuelve la fuerza del Laplaciano sin sumar la imagen original.
    Ejemplo corto: una zona plana da casi 0; un cambio brusco genera una respuesta alta.
    """
    alto, ancho = imagen.shape
    kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    buffer = np.zeros((alto, ancho), dtype=np.float64)
    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            acumulador = 0.0
            for mf in range(3):
                for mc in range(3):
                    acumulador += int(imagen[fila - 1 + mf, columna - 1 + mc]) * kernel[mf][mc]
            buffer[fila, columna] = acumulador
    return normalizar_matriz_uint8(matriz_absoluta_manual(buffer))


def filtro_roberts(imagen):
    """
    Acentuado Roberts.
    Kernels cruzados 2x2:
        Gx = [[1, 0],    Gy = [[0,  1],
              [0,-1]]          [-1,  0]]
    Magnitud G = sqrt(Gx^2 + Gy^2)
    Salida = original + G  (recortado a [0, 255])
    Ejemplo corto: si una diagonal pasa de oscuro a claro, el borde se refuerza.
    """
    print("[modelo] Aplicando filtro Roberts (acentuado)...")
    if imagen.ndim == 3:
        imagen = convertir_a_grises(imagen)

    alto, ancho = imagen.shape
    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    for fila in range(alto - 1):
        for columna in range(ancho - 1):
            gx = int(imagen[fila, columna]) - int(imagen[fila + 1, columna + 1])
            gy = int(imagen[fila, columna + 1]) - int(imagen[fila + 1, columna])
            gradiente = math.sqrt(gx * gx + gy * gy)
            valor = int(imagen[fila, columna]) + int(gradiente)
            if valor < 0:
                valor = 0
            elif valor > 255:
                valor = 255
            resultado[fila, columna] = valor
        resultado[fila, ancho - 1] = imagen[fila, ancho - 1]

    for columna in range(ancho):
        resultado[alto - 1, columna] = imagen[alto - 1, columna]

    return resultado


def filtro_prewitt(imagen):
    """
    Acentuado Prewitt.
    Kernels 3x3:
        Gx = [[-1, 0, 1],    Gy = [[-1,-1,-1],
              [-1, 0, 1],           [ 0, 0, 0],
              [-1, 0, 1]]           [ 1, 1, 1]]
    Magnitud G = sqrt(Gx^2 + Gy^2)
    Salida = original + G  (recortado a [0, 255])
    Ejemplo corto: una línea vertical activa más Gx; una horizontal activa más Gy.
    """
    print("[modelo] Aplicando filtro Prewitt (acentuado)...")
    if imagen.ndim == 3:
        imagen = convertir_a_grises(imagen)

    alto, ancho = imagen.shape
    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    Gx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    Gy = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            suma_gx = 0.0
            suma_gy = 0.0
            for mf in range(3):
                for mc in range(3):
                    pixel = int(imagen[fila - 1 + mf, columna - 1 + mc])
                    suma_gx += pixel * Gx[mf][mc]
                    suma_gy += pixel * Gy[mf][mc]
            gradiente = math.sqrt(suma_gx * suma_gx + suma_gy * suma_gy)
            valor = int(imagen[fila, columna]) + int(gradiente)
            if valor < 0:
                valor = 0
            elif valor > 255:
                valor = 255
            resultado[fila, columna] = valor

    for fila in range(alto):
        resultado[fila, 0] = imagen[fila, 0]
        resultado[fila, ancho - 1] = imagen[fila, ancho - 1]
    for columna in range(ancho):
        resultado[0, columna] = imagen[0, columna]
        resultado[alto - 1, columna] = imagen[alto - 1, columna]

    return resultado


def filtro_sobel(imagen):
    """
    Acentuado Sobel.
    Kernels 3x3:
        Gx = [[-1, 0, 1],    Gy = [[-1,-2,-1],
              [-2, 0, 2],           [ 0, 0, 0],
              [-1, 0, 1]]           [ 1, 2, 1]]
    Magnitud G = sqrt(Gx^2 + Gy^2)
    Salida = original + G  (recortado a [0, 255])
    Ejemplo corto: en placas suele resaltar bien letras porque suaviza un poco el ruido.
    """
    print("[modelo] Aplicando filtro Sobel (acentuado)...")
    if imagen.ndim == 3:
        imagen = convertir_a_grises(imagen)

    alto, ancho = imagen.shape
    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            suma_gx = 0.0
            suma_gy = 0.0
            for mf in range(3):
                for mc in range(3):
                    pixel = int(imagen[fila - 1 + mf, columna - 1 + mc])
                    suma_gx += pixel * Gx[mf][mc]
                    suma_gy += pixel * Gy[mf][mc]
            gradiente = math.sqrt(suma_gx * suma_gx + suma_gy * suma_gy)
            valor = int(imagen[fila, columna]) + int(gradiente)
            if valor < 0:
                valor = 0
            elif valor > 255:
                valor = 255
            resultado[fila, columna] = valor

    for fila in range(alto):
        resultado[fila, 0] = imagen[fila, 0]
        resultado[fila, ancho - 1] = imagen[fila, ancho - 1]
    for columna in range(ancho):
        resultado[0, columna] = imagen[0, columna]
        resultado[alto - 1, columna] = imagen[alto - 1, columna]

    return resultado


def filtro_laplaciano(imagen):
    """
    Acentuado Laplaciano.
    Kernel 3x3:
        [[-1,-1,-1],
         [-1, 8,-1],
         [-1,-1,-1]]
    Respuesta L puede ser negativa o positiva.
    Salida = original + L  (recortado a [0, 255])
    Ejemplo corto: si el centro es distinto de todos sus vecinos, se realza bastante.
    """
    print("[modelo] Aplicando filtro Laplaciano (acentuado)...")
    if imagen.ndim == 3:
        imagen = convertir_a_grises(imagen)

    alto, ancho = imagen.shape
    kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            acumulador = 0.0
            for mf in range(3):
                for mc in range(3):
                    pixel = int(imagen[fila - 1 + mf, columna - 1 + mc])
                    acumulador += pixel * kernel[mf][mc]
            valor = int(imagen[fila, columna]) + int(acumulador)
            if valor < 0:
                valor = 0
            elif valor > 255:
                valor = 255
            resultado[fila, columna] = valor

    for fila in range(alto):
        resultado[fila, 0] = imagen[fila, 0]
        resultado[fila, ancho - 1] = imagen[fila, ancho - 1]
    for columna in range(ancho):
        resultado[0, columna] = imagen[0, columna]
        resultado[alto - 1, columna] = imagen[alto - 1, columna]

    return resultado


def filtro_pasa_alto(imagen):
    """
    Devuelve la respuesta pasa-alto pura para resaltar cambios bruscos.
    Ejemplo corto: desaparece gran parte del fondo y permanecen bordes/texturas.
    """
    print("[modelo] Aplicando pasa-alto espacial...")
    if imagen.ndim == 3:
        imagen = convertir_a_grises(imagen)

    alto, ancho = imagen.shape
    kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    respuesta = np.zeros((alto, ancho), dtype=np.float64)

    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            acumulador = 0.0
            for mf in range(3):
                for mc in range(3):
                    pixel = int(imagen[fila - 1 + mf, columna - 1 + mc])
                    acumulador += pixel * kernel[mf][mc]
            respuesta[fila, columna] = acumulador

    return normalizar_matriz_uint8(matriz_absoluta_manual(respuesta))


def filtro_high_boost(imagen, factor=2.0):
    """
    Realce high-boost basado en Laplaciano: f + k*L(f).
    factor=1 deja la imagen igual; factor=2 equivale a f + L(f).
    Ejemplo corto: subir el factor hace más agresivo el borde de letras y números.
    """
    print(f"[modelo] Aplicando high-boost espacial con A={factor:.1f}...")
    if imagen.ndim == 3:
        imagen = convertir_a_grises(imagen)

    if factor < 1.0:
        factor = 1.0

    alto, ancho = imagen.shape
    kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            laplaciano = 0.0
            for mf in range(3):
                for mc in range(3):
                    pixel = int(imagen[fila - 1 + mf, columna - 1 + mc])
                    laplaciano += pixel * kernel[mf][mc]

            valor = int(imagen[fila, columna]) + (factor - 1.0) * laplaciano
            if valor < 0:
                valor = 0
            elif valor > 255:
                valor = 255
            resultado[fila, columna] = int(valor)

    for fila in range(alto):
        resultado[fila, 0] = imagen[fila, 0]
        resultado[fila, ancho - 1] = imagen[fila, ancho - 1]
    for columna in range(ancho):
        resultado[0, columna] = imagen[0, columna]
        resultado[alto - 1, columna] = imagen[alto - 1, columna]

    return resultado


def gradientes_bordes(imagen):
    """
    Devuelve los gradientes puros de cada operador.
    Ejemplo corto: sirve para comparar Roberts, Sobel o Kirsch sobre la misma placa.
    """
    print("[modelo] Calculando gradientes puros (pasa-altas espacial)...")
    if imagen.ndim == 3:
        base = convertir_a_grises(imagen)
    else:
        base = imagen
    return {
        "roberts_grad":    _gradiente_roberts_puro(base),
        "prewitt_grad":    _gradiente_prewitt_puro(base),
        "sobel_grad":      _gradiente_sobel_puro(base),
        "kirsch_grad":     _gradiente_kirsch_puro(base),
        "laplaciano_grad": _laplaciano_puro(base),
    }


def diagnostico_gradiente(imagen, operador):
    """Devuelve componentes visuales del operador seleccionado."""
    print(f"[modelo] Armando diagnostico de gradiente: {operador}...")
    if imagen.ndim == 3:
        base = convertir_a_grises(imagen)
    else:
        base = imagen

    if operador == "Roberts":
        gx, gy, magnitud = _componentes_roberts(base)
        return {
            "titulos": ["Gx (diagonal \\)", "Gy (diagonal /)", "Magnitud (G)"],
            "imagenes": [gx, gy, magnitud],
        }
    if operador == "Prewitt":
        gx, gy, magnitud = _componentes_prewitt(base)
        return {
            "titulos": ["Gx: cambio X / borde vertical", "Gy: cambio Y / borde horizontal", "Magnitud (G)"],
            "imagenes": [gx, gy, magnitud],
        }
    if operador == "Sobel":
        gx, gy, magnitud = _componentes_sobel(base)
        return {
            "titulos": ["Gx: cambio X / borde vertical", "Gy: cambio Y / borde horizontal", "Magnitud (G)"],
            "imagenes": [gx, gy, magnitud],
        }
    if operador == "Kirsch":
        norte, este, maximo = _componentes_kirsch(base)
        return {
            "titulos": ["Kirsch 0°", "Kirsch 90°", "Respuesta máxima"],
            "imagenes": [norte, este, maximo],
        }

    entrada, respuesta, respuesta_abs = _componentes_laplaciano(base)
    return {
        "titulos": ["Entrada binaria", "Respuesta L", "|L| usado"],
        "imagenes": [entrada, respuesta, respuesta_abs],
    }


def _componentes_roberts(imagen):
    alto, ancho = imagen.shape
    gx = np.zeros((alto, ancho), dtype=np.float64)
    gy = np.zeros((alto, ancho), dtype=np.float64)
    magnitud = np.zeros((alto, ancho), dtype=np.uint8)

    for fila in range(alto - 1):
        for columna in range(ancho - 1):
            valor_gx = int(imagen[fila, columna]) - int(imagen[fila + 1, columna + 1])
            valor_gy = int(imagen[fila, columna + 1]) - int(imagen[fila + 1, columna])
            gx[fila, columna] = valor_gx
            gy[fila, columna] = valor_gy
            magnitud[fila, columna] = min(int(math.sqrt(valor_gx * valor_gx + valor_gy * valor_gy)), 255)

    return normalizar_matriz_uint8(matriz_absoluta_manual(gx)), normalizar_matriz_uint8(matriz_absoluta_manual(gy)), magnitud


def _componentes_prewitt(imagen):
    alto, ancho = imagen.shape
    gx = np.zeros((alto, ancho), dtype=np.float64)
    gy = np.zeros((alto, ancho), dtype=np.float64)
    magnitud = np.zeros((alto, ancho), dtype=np.uint8)
    kernel_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    kernel_y = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            suma_gx = 0.0
            suma_gy = 0.0
            for mf in range(3):
                for mc in range(3):
                    pixel = int(imagen[fila - 1 + mf, columna - 1 + mc])
                    suma_gx += pixel * kernel_x[mf][mc]
                    suma_gy += pixel * kernel_y[mf][mc]
            gx[fila, columna] = suma_gx
            gy[fila, columna] = suma_gy
            magnitud[fila, columna] = min(int(math.sqrt(suma_gx * suma_gx + suma_gy * suma_gy)), 255)

    return normalizar_matriz_uint8(matriz_absoluta_manual(gx)), normalizar_matriz_uint8(matriz_absoluta_manual(gy)), magnitud


def _componentes_sobel(imagen):
    alto, ancho = imagen.shape
    gx = np.zeros((alto, ancho), dtype=np.float64)
    gy = np.zeros((alto, ancho), dtype=np.float64)
    magnitud = np.zeros((alto, ancho), dtype=np.uint8)
    kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            suma_gx = 0.0
            suma_gy = 0.0
            for mf in range(3):
                for mc in range(3):
                    pixel = int(imagen[fila - 1 + mf, columna - 1 + mc])
                    suma_gx += pixel * kernel_x[mf][mc]
                    suma_gy += pixel * kernel_y[mf][mc]
            gx[fila, columna] = suma_gx
            gy[fila, columna] = suma_gy
            magnitud[fila, columna] = min(int(math.sqrt(suma_gx * suma_gx + suma_gy * suma_gy)), 255)

    return normalizar_matriz_uint8(matriz_absoluta_manual(gx)), normalizar_matriz_uint8(matriz_absoluta_manual(gy)), magnitud


def _kernels_kirsch():
    return [
        [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
        [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
        [[5, -3, -3], [5, 0, -3], [5, -3, -3]],
        [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
    ]


def _componentes_kirsch(imagen):
    alto, ancho = imagen.shape
    respuesta_0 = np.zeros((alto, ancho), dtype=np.float64)
    respuesta_90 = np.zeros((alto, ancho), dtype=np.float64)
    maximo = np.zeros((alto, ancho), dtype=np.uint8)
    kernels = _kernels_kirsch()

    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            respuestas = []
            for kernel in kernels:
                acumulador = 0.0
                for mf in range(3):
                    for mc in range(3):
                        pixel = int(imagen[fila - 1 + mf, columna - 1 + mc])
                        acumulador += pixel * kernel[mf][mc]
                respuestas.append(acumulador)

            respuesta_0[fila, columna] = respuestas[0]
            respuesta_90[fila, columna] = respuestas[2]
            max_resp = max(abs(valor) for valor in respuestas)
            maximo[fila, columna] = min(int(max_resp), 255)

    return (
        normalizar_matriz_uint8(matriz_absoluta_manual(respuesta_0)),
        normalizar_matriz_uint8(matriz_absoluta_manual(respuesta_90)),
        maximo,
    )


def _componentes_laplaciano(imagen):
    alto, ancho = imagen.shape
    kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    respuesta = np.zeros((alto, ancho), dtype=np.float64)

    for fila in range(1, alto - 1):
        for columna in range(1, ancho - 1):
            acumulador = 0.0
            for mf in range(3):
                for mc in range(3):
                    acumulador += int(imagen[fila - 1 + mf, columna - 1 + mc]) * kernel[mf][mc]
            respuesta[fila, columna] = acumulador

    return (
        imagen,
        visualizar_laplaciano_firmado(respuesta),
        normalizar_matriz_uint8(matriz_absoluta_manual(respuesta)),
    )


def visualizar_laplaciano_firmado(respuesta):
    """
    Muestra valores negativos y positivos alrededor de gris medio.
    Así una zona sin cambio queda gris, un cambio positivo claro y uno negativo oscuro.
    """
    alto, ancho = respuesta.shape
    maximo_abs = 0.0

    for fila in range(alto):
        for columna in range(ancho):
            valor = float(respuesta[fila, columna])
            if valor < 0:
                valor = -valor
            if valor > maximo_abs:
                maximo_abs = valor

    resultado = np.zeros((alto, ancho), dtype=np.uint8)
    if maximo_abs == 0:
        for fila in range(alto):
            for columna in range(ancho):
                resultado[fila, columna] = 128
        return resultado

    for fila in range(alto):
        for columna in range(ancho):
            valor = 128.0 + (float(respuesta[fila, columna]) / maximo_abs) * 127.0
            if valor < 0:
                valor = 0
            elif valor > 255:
                valor = 255
            resultado[fila, columna] = int(valor)

    return resultado

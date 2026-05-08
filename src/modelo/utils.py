"""Utilidades compartidas por los algoritmos manuales."""

import numpy as np

def aplicar_por_canal(imagen, funcion_canal):
    """Aplica una función a cada canal y vuelve a armar la imagen RGB."""
    if imagen.ndim == 2:
        return funcion_canal(imagen)

    alto, ancho, canales = imagen.shape
    resultado = np.zeros((alto, ancho, canales), dtype=np.uint8)

    for canal in range(canales):
        canal_original = np.zeros((alto, ancho), dtype=np.uint8)
        for fila in range(alto):
            for columna in range(ancho):
                canal_original[fila, columna] = imagen[fila, columna, canal]

        canal_filtrado = funcion_canal(canal_original)
        for fila in range(alto):
            for columna in range(ancho):
                resultado[fila, columna, canal] = canal_filtrado[fila, columna]

    return resultado

def normalizar_matriz_uint8(matriz):
    """Normaliza cualquier matriz real al rango [0, 255]."""
    alto, ancho = matriz.shape
    minimo = float(matriz[0, 0])
    maximo = minimo

    for fila in range(alto):
        for columna in range(ancho):
            valor = float(matriz[fila, columna])
            if valor < minimo:
                minimo = valor
            if valor > maximo:
                maximo = valor

    if maximo == minimo:
        return np.zeros(matriz.shape, dtype=np.uint8)

    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            valor = (float(matriz[fila, columna]) - minimo) * 255.0 / (maximo - minimo)
            if valor < 0:
                valor = 0
            elif valor > 255:
                valor = 255
            resultado[fila, columna] = int(valor)

    return resultado


def matriz_absoluta_manual(matriz):
    """Calcula valor absoluto elemento por elemento."""
    alto, ancho = matriz.shape
    resultado = np.zeros((alto, ancho), dtype=np.float64)

    for fila in range(alto):
        for columna in range(ancho):
            valor = float(matriz[fila, columna])
            if valor < 0:
                valor = -valor
            resultado[fila, columna] = valor

    return resultado


def parte_real_manual(matriz_compleja):
    """Extrae la parte real de una matriz compleja con un recorrido explícito."""
    alto, ancho = matriz_compleja.shape
    resultado = np.zeros((alto, ancho), dtype=np.float64)

    for fila in range(alto):
        for columna in range(ancho):
            resultado[fila, columna] = matriz_compleja[fila, columna].real

    return resultado


def diferencia_absoluta_manual(imagen_a, imagen_b):
    """
    Genera el "mapa de cambio" entre dos imágenes del mismo tamaño.

    Idea del mapa:
    - Se toma una imagen de referencia y otra imagen resultado.
    - En este proyecto normalmente se compara:
      imagen con ruido vs imagen filtrada.
    - Para cada píxel se calcula la diferencia absoluta:
      |pixel_original - pixel_resultado|

    Cómo se interpreta:
    - Valor pequeño  -> el filtro casi no modificó ese píxel.
    - Valor grande   -> el filtro sí alteró notablemente ese píxel.
    - Zonas oscuras  -> pocos cambios.
    - Zonas claras   -> cambios fuertes.

    Para qué sirve en la práctica:
    - Visualizar dónde actuó más el filtro espacial.
    - Comparar qué tan agresivos son media, mediana y moda.
    - Ver si el filtrado realmente corrigió regiones con ruido
      sal y pimienta o si también afectó zonas que estaban bien.

    Nota:
    Este mapa no es una imagen "mejorada" final, sino una imagen
    de diagnostico. Su objetivo es explicar el efecto del filtro,
    no reemplazar el resultado filtrado.
    """
    print("[modelo] Calculando mapa de cambio...")
    if imagen_a.shape != imagen_b.shape:
        raise ValueError("Las imágenes deben tener el mismo tamaño para calcular la diferencia.")

    if imagen_a.ndim == 2:
        alto, ancho = imagen_a.shape
        resultado = np.zeros((alto, ancho), dtype=np.uint8)
        for fila in range(alto):
            for columna in range(ancho):
                diferencia = int(imagen_a[fila, columna]) - int(imagen_b[fila, columna])
                if diferencia < 0:
                    diferencia = -diferencia
                resultado[fila, columna] = diferencia
        return resultado

    alto, ancho, canales = imagen_a.shape
    resultado = np.zeros((alto, ancho, canales), dtype=np.uint8)
    for fila in range(alto):
        for columna in range(ancho):
            for canal in range(canales):
                diferencia = int(imagen_a[fila, columna, canal]) - int(imagen_b[fila, columna, canal])
                if diferencia < 0:
                    diferencia = -diferencia
                resultado[fila, columna, canal] = diferencia
    return resultado

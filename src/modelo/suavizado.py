"""Filtros de suavizado en dominio espacial."""

import numpy as np

from .utils import aplicar_por_canal

def crear_mascara_media(tamano_mascara):
    """
    Construye una máscara de promedio NxN.
    Ejemplo corto: en 3x3 cada celda vale 1/9, así todos los vecinos aportan
    el mismo peso al nuevo píxel central.
    """
    mascara = np.zeros((tamano_mascara, tamano_mascara), dtype=np.float64)
    valor = 1.0 / (tamano_mascara * tamano_mascara)

    for fila in range(tamano_mascara):
        for columna in range(tamano_mascara):
            mascara[fila, columna] = valor

    return mascara


def convolucionar_manual_grises(imagen, mascara):
    """
    Recorre una imagen en grises usando una máscara NxN.
    La máscara debe ser impar para tener un píxel central claro.
    Ejemplo corto: una ventana 3x3 se alinea sobre el píxel actual, multiplica
    vecino por peso y suma todo en un solo valor.
    """
    alto, ancho = imagen.shape
    kh, kw = mascara.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.zeros((alto + 2 * pad_h, ancho + 2 * pad_w), dtype=np.float64)
    for fila in range(alto):
        for columna in range(ancho):
            padded[fila + pad_h, columna + pad_w] = float(imagen[fila, columna])

    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            acumulador = 0.0
            for mf in range(kh):
                for mc in range(kw):
                    pixel = padded[fila + mf, columna + mc]
                    peso = mascara[mf, mc]
                    acumulador += pixel * peso

            valor = int(round(acumulador))
            if valor < 0:
                valor = 0
            elif valor > 255:
                valor = 255
            resultado[fila, columna] = valor

    return resultado


def filtro_media(imagen, tamano_mascara):
    """Suaviza promediando vecinos; útil para ruido suave, pero puede borrar bordes."""
    print(f"[modelo] Aplicando filtro de media con mascara {tamano_mascara}x{tamano_mascara}...")
    mascara = crear_mascara_media(tamano_mascara)
    return aplicar_por_canal(imagen, lambda canal: convolucionar_manual_grises(canal, mascara))


def filtro_mediana(imagen, tamano_mascara):
    """Toma el valor central al ordenar la ventana; suele limpiar bien sal y pimienta."""
    print(f"[modelo] Aplicando filtro de mediana con mascara {tamano_mascara}x{tamano_mascara}...")
    return aplicar_por_canal(imagen, lambda canal: filtro_mediana_grises(canal, tamano_mascara))


def filtro_moda(imagen, tamano_mascara):
    """Conserva el valor más repetido de la ventana; ayuda cuando hay tonos dominantes."""
    print(f"[modelo] Aplicando filtro de moda con mascara {tamano_mascara}x{tamano_mascara}...")
    return aplicar_por_canal(imagen, lambda canal: filtro_moda_grises(canal, tamano_mascara))

def filtro_mediana_grises(imagen, tamano_mascara):
    """
    Filtro de mediana con ordenamiento manual por inserción.
    Ejemplo corto: en [0, 120, 121, 122, 255] la mediana es 121, por eso
    ignora impulsos extremos sin inventar un promedio extraño.
    """
    alto, ancho = imagen.shape
    pad = tamano_mascara // 2
    padded = np.zeros((alto + 2 * pad, ancho + 2 * pad), dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            padded[fila + pad, columna + pad] = imagen[fila, columna]

    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            ventana = []
            for mf in range(tamano_mascara):
                for mc in range(tamano_mascara):
                    ventana.append(int(padded[fila + mf, columna + mc]))

            for indice in range(1, len(ventana)):
                clave = ventana[indice]
                posicion = indice - 1
                while posicion >= 0 and ventana[posicion] > clave:
                    ventana[posicion + 1] = ventana[posicion]
                    posicion -= 1
                ventana[posicion + 1] = clave

            resultado[fila, columna] = ventana[len(ventana) // 2]

    return resultado


def filtro_moda_grises(imagen, tamano_mascara):
    """
    Filtro de moda con conteo manual de frecuencias.
    Ejemplo corto: si en la ventana aparece 180 cinco veces y 0 una vez,
    se conserva 180 porque describe mejor el vecindario.
    """
    alto, ancho = imagen.shape
    pad = tamano_mascara // 2
    padded = np.zeros((alto + 2 * pad, ancho + 2 * pad), dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            padded[fila + pad, columna + pad] = imagen[fila, columna]

    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            frecuencias = {}
            primer_valor = int(padded[fila, columna])
            valor_moda = primer_valor
            frecuencia_moda = 0

            for mf in range(tamano_mascara):
                for mc in range(tamano_mascara):
                    valor = int(padded[fila + mf, columna + mc])
                    if valor in frecuencias:
                        frecuencias[valor] += 1
                    else:
                        frecuencias[valor] = 1

                    if frecuencias[valor] > frecuencia_moda:
                        frecuencia_moda = frecuencias[valor]
                        valor_moda = valor

            resultado[fila, columna] = valor_moda

    return resultado

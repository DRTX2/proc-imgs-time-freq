"""Carga, conversión, histograma y ruido básico."""

import random

import cv2
import numpy as np

def cargar_imagen(ruta):
    """Carga una imagen RGB desde disco."""
    print(f"[modelo] Cargando imagen: {ruta}")
    datos = np.fromfile(str(ruta), np.uint8)
    img_bgr = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"No se pudo cargar: {ruta}")
    imagen_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    alto, ancho = imagen_rgb.shape[:2]
    print(f"[modelo] Imagen lista: {ancho}x{alto}")
    return imagen_rgb


def convertir_a_grises(imagen_rgb):
    """
    Convierte RGB a grises con una pasada pixel a pixel.
    Ejemplo corto: un rojo puro pesa más por R, pero el verde aporta todavía
    más porque el ojo humano percibe mejor esa intensidad.
    """
    print("[modelo] Pasando a escala de grises...")
    alto, ancho = imagen_rgb.shape[:2]
    gris = np.zeros((alto, ancho), dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            r = int(imagen_rgb[fila, columna, 0])
            g = int(imagen_rgb[fila, columna, 1])
            b = int(imagen_rgb[fila, columna, 2])
            valor = int(0.299 * r + 0.587 * g + 0.114 * b)
            if valor < 0:
                valor = 0
            elif valor > 255:
                valor = 255
            gris[fila, columna] = valor

    return gris


def normalizar_histograma_grises(imagen_gris):
    """
    Ecualiza el histograma con la CDF para estirar el contraste disponible.
    Ejemplo corto: si casi todo cae entre 90 y 140, se reparte hacia 0..255
    para que detalles débiles no queden escondidos.
    """
    print("[modelo] Ecualizando histograma...")
    alto, ancho = imagen_gris.shape
    histograma = [0] * 256

    for fila in range(alto):
        for columna in range(ancho):
            histograma[int(imagen_gris[fila, columna])] += 1

    acumulado = [0] * 256
    acumulado[0] = histograma[0]
    for indice in range(1, 256):
        acumulado[indice] = acumulado[indice - 1] + histograma[indice]

    cdf_min = 0
    for valor in acumulado:
        if valor > 0:
            cdf_min = valor
            break

    total = alto * ancho
    resultado = np.zeros((alto, ancho), dtype=np.uint8)
    if total == cdf_min:
        return imagen_gris.copy()

    for fila in range(alto):
        for columna in range(ancho):
            pixel = int(imagen_gris[fila, columna])
            nuevo = round((acumulado[pixel] - cdf_min) * 255 / (total - cdf_min))
            if nuevo < 0:
                nuevo = 0
            elif nuevo > 255:
                nuevo = 255
            resultado[fila, columna] = nuevo

    return resultado


def calcular_histograma_grises(imagen_gris):
    """Cuenta manualmente las frecuencias de intensidad de una imagen en grises."""
    alto, ancho = imagen_gris.shape
    histograma = [0] * 256

    for fila in range(alto):
        for columna in range(ancho):
            histograma[int(imagen_gris[fila, columna])] += 1

    return histograma


def calcular_tamano_mascara_maximo(imagen):
    """
    Calcula el tamaño máximo permitido para la máscara espacial.
    Se limita a kernels impares razonables para mantener fluido el algoritmo manual.
    """
    alto, ancho = imagen.shape[:2]
    lado_menor = min(alto, ancho)
    maximo = int(lado_menor * 0.12)
    if maximo % 2 == 0:
        maximo -= 1
    maximo = min(maximo, 25)
    return max(3, maximo)


def agregar_ruido_sal_pimienta(imagen, intensidad):
    """
    Agrega ruido impulsivo sal y pimienta con un recorrido por posiciones.
    Ejemplo corto: con intensidad 0.10 en 1000 pixeles se alteran unos 100,
    mitad blancos (sal) y mitad negros (pimienta).
    """
    print(f"[modelo] Agregando ruido sal y pimienta: {int(round(intensidad * 100))}%")
    resultado = imagen.copy()
    alto, ancho = resultado.shape[:2]
    total_pixeles = alto * ancho
    total_ruido = int(total_pixeles * intensidad)
    total_sal = total_ruido // 2
    total_pimienta = total_ruido - total_sal

    _aplicar_ruido(resultado, total_sal, 255)
    _aplicar_ruido(resultado, total_pimienta, 0)

    return resultado


def _aplicar_ruido(imagen, cantidad, valor):
    alto, ancho = imagen.shape[:2]

    for _ in range(cantidad):
        fila = random.randint(0, alto - 1)
        columna = random.randint(0, ancho - 1)

        if imagen.ndim == 2:
            imagen[fila, columna] = valor
        else:
            for canal in range(imagen.shape[2]):
                imagen[fila, columna, canal] = valor

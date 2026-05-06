import math
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
    """Convierte RGB a grises usando una pasada pixel a pixel."""
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
    """Ecualiza el histograma de una imagen en grises con conteo manual."""
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


def binarizar_imagen(imagen_gris, umbral):
    """Binariza manualmente una imagen en grises."""
    print(f"[modelo] Binarizando con umbral {umbral}...")
    alto, ancho = imagen_gris.shape
    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            if int(imagen_gris[fila, columna]) >= umbral:
                resultado[fila, columna] = 255
            else:
                resultado[fila, columna] = 0

    return resultado


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
    intensidad: proporción de pixeles afectados (0.0 a 1.0).
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


def crear_mascara_media(tamano_mascara):
    """Construye manualmente una máscara de promedio NxN."""
    mascara = np.zeros((tamano_mascara, tamano_mascara), dtype=np.float64)
    valor = 1.0 / (tamano_mascara * tamano_mascara)

    for fila in range(tamano_mascara):
        for columna in range(tamano_mascara):
            mascara[fila, columna] = valor

    return mascara


def convolucionar_manual_grises(imagen, mascara):
    """
    Recorre una imagen en grises usando una máscara NxN.
    La máscara debe ser impar para tener un píxel central.
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
    """Filtro de media aplicado por canal cuando la imagen es RGB."""
    # Suaviza promediando los valores de cada vecindad.
    print(f"[modelo] Aplicando filtro de media con mascara {tamano_mascara}x{tamano_mascara}...")
    mascara = crear_mascara_media(tamano_mascara)
    return aplicar_por_canal(imagen, lambda canal: convolucionar_manual_grises(canal, mascara))


def filtro_mediana(imagen, tamano_mascara):
    """Filtro de mediana aplicado por canal."""
    # Reduce ruido impulsivo sin depender de una funcion de mediana externa.
    print(f"[modelo] Aplicando filtro de mediana con mascara {tamano_mascara}x{tamano_mascara}...")
    return aplicar_por_canal(imagen, lambda canal: filtro_mediana_grises(canal, tamano_mascara))


def filtro_moda(imagen, tamano_mascara):
    """Filtro de moda aplicado por canal."""
    # En imagenes binarias conserva el valor predominante de la vecindad.
    print(f"[modelo] Aplicando filtro de moda con mascara {tamano_mascara}x{tamano_mascara}...")
    return aplicar_por_canal(imagen, lambda canal: filtro_moda_grises(canal, tamano_mascara))


def aplicar_por_canal(imagen, funcion_canal):
    """Aplica una función a cada canal si la imagen es RGB."""
    if imagen.ndim == 2:
        return funcion_canal(imagen)

    alto, ancho, canales = imagen.shape
    resultado = np.zeros((alto, ancho, canales), dtype=np.uint8)

    for canal in range(canales):
        resultado[:, :, canal] = funcion_canal(imagen[:, :, canal])

    return resultado


def filtro_mediana_grises(imagen, tamano_mascara):
    """
    Filtro de mediana con ordenamiento manual por inserción.
    No usa numpy.median() ni sorted().
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
    """Filtro de moda con conteo manual de frecuencias."""
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


def crear_filtro_gaussiano(altura, ancho, d0):
    """
    Crea una máscara gaussiana para filtro pasabajas en frecuencia.
    """
    if d0 <= 0:
        d0 = 1.0

    cy = altura // 2
    cx = ancho // 2
    filtro = np.zeros((altura, ancho), dtype=np.float64)

    for fila in range(altura):
        for columna in range(ancho):
            distancia = math.sqrt((fila - cy) ** 2 + (columna - cx) ** 2)
            filtro[fila, columna] = math.exp(-(distancia ** 2) / (2.0 * d0 ** 2))

    return filtro


def aplicar_filtro_frecuencia(espectro, filtro):
    """Multiplica el espectro complejo por la máscara usando loops."""
    alto, ancho = espectro.shape
    resultado = np.zeros_like(espectro)

    for fila in range(alto):
        for columna in range(ancho):
            resultado[fila, columna] = espectro[fila, columna] * filtro[fila, columna]

    return resultado


def espectro_log(espectro_complejo):
    """Convierte un espectro complejo a magnitud logarítmica para visualización."""
    magnitud = np.log1p(np.abs(espectro_complejo))
    return normalizar_matriz_uint8(magnitud)


def normalizar_matriz_uint8(matriz):
    """Normaliza cualquier matriz real al rango [0, 255]."""
    minimo = float(matriz.min())
    maximo = float(matriz.max())

    if maximo == minimo:
        return np.zeros(matriz.shape, dtype=np.uint8)

    alto, ancho = matriz.shape
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


def filtrar_frecuencia_matriz(imagen, d0):
    """Aplica el filtrado gaussiano a una matriz 2D y devuelve también sus diagnósticos."""
    alto, ancho = imagen.shape
    espectro = np.fft.fft2(imagen.astype(np.float64))
    espectro_centrado = np.fft.fftshift(espectro)
    mascara = crear_filtro_gaussiano(alto, ancho, d0)
    espectro_filtrado = aplicar_filtro_frecuencia(espectro_centrado, mascara)
    reconstruida = np.fft.ifft2(np.fft.ifftshift(espectro_filtrado))
    resultado = normalizar_matriz_uint8(np.real(reconstruida))
    return resultado, espectro_centrado, espectro_filtrado, mascara


def fourier_filtrar_canal(imagen, d0):
    """Aplica filtrado gaussiano pasabajas a un canal."""
    resultado, _, _, _ = filtrar_frecuencia_matriz(imagen, d0)
    return resultado


def filtro_frecuencia_gaussiano(imagen, d0):
    """Aplica el filtro de frecuencia a una imagen en gris o RGB."""
    # El filtrado usa FFT de NumPy, pero la mascara se construye en este modulo.
    print(f"[modelo] Aplicando filtro gaussiano en frecuencia con D0={d0}...")
    return aplicar_por_canal(imagen, lambda canal: fourier_filtrar_canal(canal, d0))


def diagnostico_frecuencia(imagen, d0):
    """Genera imágenes de apoyo para la pestaña de frecuencia."""
    print("[modelo] Armando diagnostico de frecuencia...")
    if imagen.ndim == 3:
        base = convertir_a_grises(imagen)
    else:
        base = imagen

    resultado, espectro_original, espectro_filtrado, mascara = filtrar_frecuencia_matriz(base, d0)
    return {
        "base_gris": base,
        "resultado_gris": resultado,
        "espectro_original": espectro_log(espectro_original),
        "mascara": normalizar_matriz_uint8(mascara),
        "espectro_filtrado": espectro_log(espectro_filtrado),
    }


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

import numpy as np
import cv2
import math


def cargar_imagen(ruta):
    """Carga una imagen RGB desde disco."""
    datos = np.fromfile(str(ruta), np.uint8)
    img_bgr = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"No se pudo cargar: {ruta}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def agregar_ruido_sal_pimienta(imagen, intensidad):
    """
    Agrega ruido impulsivo sal y pimienta.
    intensidad: proporcion de pixeles afectados (0.0 a 1.0).
    La mitad seran sal (255) y la otra mitad pimienta (0).
    """
    resultado = imagen.copy()
    alto, ancho = resultado.shape[:2]
    total = alto * ancho

    n_sal = int(total * intensidad / 2)
    n_pim = int(total * intensidad / 2)

    # Sal (blanco)
    fs = np.random.randint(0, alto, n_sal)
    cs = np.random.randint(0, ancho, n_sal)
    resultado[fs, cs] = 255

    # Pimienta (negro)
    fp = np.random.randint(0, alto, n_pim)
    cp = np.random.randint(0, ancho, n_pim)
    resultado[fp, cp] = 0

    return resultado


def normalizar_histograma_rgb(imagen_rgb):
    """
    Normaliza el histograma de la imagen RGB aplicando expansión min-max
    por cada canal independientemente, estirando los valores al rango [0, 255].
    """
    resultado = np.zeros_like(imagen_rgb)
    for c in range(3):
        canal = imagen_rgb[:, :, c].astype(np.float32)
        vmin = canal.min()
        vmax = canal.max()
        if vmax > vmin:
            canal_norm = (canal - vmin) * 255.0 / (vmax - vmin)
        else:
            canal_norm = canal
        resultado[:, :, c] = np.clip(canal_norm, 0, 255).astype(np.uint8)
    return resultado


def convertir_a_grises(imagen_rgb):
    """RGB a escala de grises con pesos perceptuales Luma BT.601."""
    r = imagen_rgb[:, :, 0].astype(np.float64)
    g = imagen_rgb[:, :, 1].astype(np.float64)
    b = imagen_rgb[:, :, 2].astype(np.float64)
    gris = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gris, 0, 255).astype(np.uint8)


def binarizar_imagen(imagen_grises, threshold):
    """Binariza: pixel >= threshold se vuelve 255, sino 0."""
    return np.where(imagen_grises >= threshold, 255, 0).astype(np.uint8)


def calcular_tamano_mascara_maximo(imagen):
    """
    Calcula el tamaño maximo permitido para la mascara espacial.
    Es el 60% del lado menor de la imagen, redondeado al impar inferior.
    """
    alto, ancho = imagen.shape[:2]
    lado_menor = min(alto, ancho)
    maximo = int(lado_menor * 0.6)
    if maximo % 2 == 0:
        maximo -= 1
    return max(3, maximo)


def convolucionar_manual(imagen, mascara):
    """
    Convolución 2D con loops manuales.
    Aplica padding con ceros para preservar el tamaño original.
    No usa scipy.ndimage.convolve() ni cv2.filter2D().
    """
    alto, ancho = imagen.shape
    kh, kw = mascara.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Imagen con padding de ceros en los bordes
    padded = np.zeros((alto + 2 * pad_h, ancho + 2 * pad_w), dtype=np.float64)
    padded[pad_h:pad_h + alto, pad_w:pad_w + ancho] = imagen.astype(np.float64)

    resultado = np.zeros((alto, ancho), dtype=np.float64)

    for i in range(alto):
        for j in range(ancho):
            acumulador = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    acumulador += padded[i + ki, j + kj] * mascara[ki, kj]
            resultado[i, j] = acumulador

    return np.clip(resultado, 0, 255).astype(np.uint8)


def filtro_media(imagen, tamano_mascara):
    """
    Filtro de media (promedio).
    Crea una mascara de unos dividida entre n*n y la convoluciona.
    """
    n = tamano_mascara
    mascara = np.ones((n, n), dtype=np.float64) / (n * n)
    return convolucionar_manual(imagen, mascara)


def filtro_mediana(imagen, tamano_mascara):
    """
    Filtro de mediana con ordenamiento manual (insertion sort).
    No usa numpy.median() ni scipy.ndimage.median_filter().
    """
    alto, ancho = imagen.shape
    pad = tamano_mascara // 2

    padded = np.zeros((alto + 2 * pad, ancho + 2 * pad), dtype=np.uint8)
    padded[pad:pad + alto, pad:pad + ancho] = imagen

    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    for i in range(alto):
        for j in range(ancho):
            # Extraer ventana manualmente
            ventana = []
            for ki in range(tamano_mascara):
                for kj in range(tamano_mascara):
                    ventana.append(int(padded[i + ki, j + kj]))

            # Ordenamiento por insercion (manual, sin sorted/numpy)
            for a in range(1, len(ventana)):
                clave = ventana[a]
                b = a - 1
                while b >= 0 and ventana[b] > clave:
                    ventana[b + 1] = ventana[b]
                    b -= 1
                ventana[b + 1] = clave

            # El valor central de la lista ordenada es la mediana
            resultado[i, j] = ventana[len(ventana) // 2]

    return resultado


def filtro_moda(imagen, tamano_mascara):
    """
    Filtro de moda: asigna el valor mas frecuente de la ventana.
    Cuenta frecuencias manualmente con diccionario, sin scipy.
    """
    alto, ancho = imagen.shape
    pad = tamano_mascara // 2

    padded = np.zeros((alto + 2 * pad, ancho + 2 * pad), dtype=np.uint8)
    padded[pad:pad + alto, pad:pad + ancho] = imagen

    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    for i in range(alto):
        for j in range(ancho):
            ventana = []
            for ki in range(tamano_mascara):
                for kj in range(tamano_mascara):
                    ventana.append(int(padded[i + ki, j + kj]))

            # Conteo de frecuencias manual
            frecuencias = {}
            for val in ventana:
                if val in frecuencias:
                    frecuencias[val] += 1
                else:
                    frecuencias[val] = 1

            # Buscar el valor con mayor frecuencia
            moda_val = ventana[0]
            max_freq = 0
            for val, freq in frecuencias.items():
                if freq > max_freq:
                    max_freq = freq
                    moda_val = val

            resultado[i, j] = moda_val

    return resultado


def crear_filtro_gaussiano(altura, ancho, sigma):
    """
    Crea mascara Gaussiana para filtro pasabajas en frecuencia.
    H(u,v) = exp(-D(u,v)^2 / (2*sigma^2))
    D(u,v) = sqrt((u - cy)^2 + (v - cx)^2)
    Construida con loops manuales.
    """
    cy = altura // 2
    cx = ancho // 2
    filtro = np.zeros((altura, ancho), dtype=np.float64)

    for u in range(altura):
        for v in range(ancho):
            d = math.sqrt((u - cy) ** 2 + (v - cx) ** 2)
            filtro[u, v] = math.exp(-(d ** 2) / (2.0 * sigma ** 2))

    return filtro


def aplicar_filtro_frecuencia(espectro, filtro):
    """
    Multiplica espectro complejo por filtro real, punto a punto.
    Implementado con loops manuales (sin operador * directo).
    """
    alto, ancho = espectro.shape
    resultado = np.zeros_like(espectro)

    for i in range(alto):
        for j in range(ancho):
            resultado[i, j] = espectro[i, j] * filtro[i, j]

    return resultado


def fourier_procesada_completa(imagen, sigma):
    """
    Pipeline completo de filtrado en frecuencia:
    1. FFT 2D (numpy.fft.fft2)
    2. Centrar espectro (fftshift)
    3. Crear mascara Gaussiana manual
    4. Multiplicar espectro x filtro manual
    5. FFT inversa (numpy.fft.ifft2)
    6. Tomar parte real, normalizar a [0, 255]

    Retorna: (imagen_resultado, espectro_original, espectro_filtrado, mascara)
    """
    alto, ancho = imagen.shape

    # Transformada directa
    espectro = np.fft.fft2(imagen.astype(np.float64))
    espectro_centrado = np.fft.fftshift(espectro)

    # Mascara gaussiana (loops manuales)
    mascara = crear_filtro_gaussiano(alto, ancho, sigma)

    # Multiplicacion manual espectro x filtro
    espectro_filtrado = aplicar_filtro_frecuencia(espectro_centrado, mascara)

    # Transformada inversa
    espectro_deshift = np.fft.ifftshift(espectro_filtrado)
    reconstruida = np.fft.ifft2(espectro_deshift)
    resultado = np.real(reconstruida)

    # Normalizar al rango [0, 255]
    vmin, vmax = resultado.min(), resultado.max()
    if vmax > vmin:
        resultado = (resultado - vmin) / (vmax - vmin) * 255.0

    return (
        np.clip(resultado, 0, 255).astype(np.uint8),
        espectro_centrado,
        espectro_filtrado,
        mascara,
    )


def espectro_log(espectro_complejo):
    """Calcula magnitud en escala log para visualizacion: log(1 + |F|)."""
    magnitud = np.abs(espectro_complejo)
    log_mag = np.log1p(magnitud)
    if log_mag.max() > 0:
        log_mag = (log_mag / log_mag.max()) * 255.0
    return log_mag.astype(np.uint8)
from pathlib import Path
import cv2
import numpy as np
from collections import Counter

# --- Configuración ---

DEFAULT_IMAGE_PATH = Path("informe/formación estelar de W51.png")


def buscar_imagen_inicial():
    """Retorna la ruta de la imagen de prueba si existe."""
    return DEFAULT_IMAGE_PATH if DEFAULT_IMAGE_PATH.exists() else None


def cargar_imagen(path=None):
    """
    Carga una imagen desde disco y la retorna en formato RGB.
    Si la ruta tiene caracteres especiales usa np.fromfile para evitar
    problemas de codificación en Windows.
    """
    if not path:
        return np.zeros((320, 480, 3), dtype=np.uint8)

    try:
        datos = np.fromfile(str(path), np.uint8)
        img_bgr = cv2.imdecode(datos, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError("No se pudo decodificar la imagen")

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    except Exception:
        return np.zeros((320, 480, 3), dtype=np.uint8)


# --- Conversión a escala de grises ---

def rgb_a_gris(img_rgb):
    """
    Convierte RGB a escala de grises con pesos perceptuales (Luma BT.601).
    Estos pesos se usan porque el ojo humano es más sensible al verde.
    """
    r = img_rgb[:, :, 0].astype(np.float64)
    g = img_rgb[:, :, 1].astype(np.float64)
    b = img_rgb[:, :, 2].astype(np.float64)

    gris = 0.299 * r + 0.587 * g + 0.114 * b

    return np.clip(gris, 0, 255).astype(np.uint8)


# --- Ruido sal y pimienta ---

def ruido_sal_pimienta(imagen_gris, probabilidad=0.05):
    """
    Agrega ruido impulsivo (sal y pimienta) a una imagen en escala de grises.
    - probabilidad: porcentaje total de pixeles afectados (mitad sal, mitad pimienta).
    """
    resultado = imagen_gris.copy()
    alto, ancho = resultado.shape
    total_pixeles = alto * ancho

    # Cantidad de pixeles que se van a modificar
    n_sal = int(total_pixeles * probabilidad / 2)
    n_pimienta = int(total_pixeles * probabilidad / 2)

    # Generamos posiciones aleatorias para la sal (blanco = 255)
    filas_sal = np.random.randint(0, alto, n_sal)
    cols_sal = np.random.randint(0, ancho, n_sal)
    resultado[filas_sal, cols_sal] = 255

    # Generamos posiciones aleatorias para la pimienta (negro = 0)
    filas_pim = np.random.randint(0, alto, n_pimienta)
    cols_pim = np.random.randint(0, ancho, n_pimienta)
    resultado[filas_pim, cols_pim] = 0

    return resultado


# --- Filtros de dominio espacial (convolución manual) ---

def _extraer_vecindario(imagen, fila, col, tam_kernel):
    """
    Extrae los pixeles vecinos alrededor de (fila, col) usando el tamaño
    del kernel. Maneja los bordes recortando las coordenadas.
    """
    mitad = tam_kernel // 2
    alto, ancho = imagen.shape

    f_ini = max(0, fila - mitad)
    f_fin = min(alto, fila + mitad + 1)
    c_ini = max(0, col - mitad)
    c_fin = min(ancho, col + mitad + 1)

    return imagen[f_ini:f_fin, c_ini:c_fin]


def filtro_media(imagen_gris, tam_kernel=3):
    """
    Filtro de media (promedio): suaviza la imagen pero puede difuminar bordes.
    Implementado con convolución manual usando un kernel de unos.
    """
    # Construimos el kernel de promedio
    kernel = np.ones((tam_kernel, tam_kernel), dtype=np.float64)
    kernel = kernel / kernel.sum()

    return _convolucion_espacial(imagen_gris, kernel)


def filtro_mediana(imagen_gris, tam_kernel=3):
    """
    Filtro de mediana: ordena los vecinos y toma el valor central.
    Muy bueno para eliminar ruido sal y pimienta sin difuminar bordes.
    """
    alto, ancho = imagen_gris.shape
    resultado = np.zeros_like(imagen_gris)

    for i in range(alto):
        for j in range(ancho):
            vecinos = _extraer_vecindario(imagen_gris, i, j, tam_kernel)
            resultado[i, j] = np.median(vecinos)

    return resultado


def filtro_moda(imagen_gris, tam_kernel=3):
    """
    Filtro de moda: asigna a cada pixel el valor más frecuente entre
    sus vecinos. Util para imagenes con pocos niveles de gris.
    """
    alto, ancho = imagen_gris.shape
    resultado = np.zeros_like(imagen_gris)

    for i in range(alto):
        for j in range(ancho):
            vecinos = _extraer_vecindario(imagen_gris, i, j, tam_kernel)
            valores = vecinos.flatten().tolist()
            # Counter.most_common(1) nos da el valor más repetido
            conteo = Counter(valores)
            resultado[i, j] = conteo.most_common(1)[0][0]

    return resultado


def _convolucion_espacial(imagen_gris, kernel):
    """
    Aplica convolución 2D manual entre la imagen y un kernel dado.
    Recorre pixel por pixel y multiplica por los pesos del kernel.
    """
    alto, ancho = imagen_gris.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Padding con ceros para manejar los bordes
    padded = np.pad(imagen_gris.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    resultado = np.zeros((alto, ancho), dtype=np.float64)

    for i in range(alto):
        for j in range(ancho):
            region = padded[i:i + kh, j:j + kw]
            resultado[i, j] = np.sum(region * kernel)

    return np.clip(resultado, 0, 255).astype(np.uint8)


# --- Dominio de frecuencia (FFT) ---

def filtro_frecuencia_pasabajas(imagen_gris, radio=30):
    """
    Aplica un filtro pasabajas ideal en el dominio de la frecuencia:
    1. Calcula la FFT 2D de la imagen
    2. Centra el espectro (shift)
    3. Crea una máscara circular con el radio dado
    4. Multiplica el espectro por la máscara (convolución en frecuencia)
    5. Aplica la FFT inversa para reconstruir la imagen filtrada
    """
    alto, ancho = imagen_gris.shape

    # Transformada de Fourier 2D
    espectro = np.fft.fft2(imagen_gris.astype(np.float64))
    espectro_centrado = np.fft.fftshift(espectro)

    # Construir mascara circular (filtro pasabajas ideal)
    centro_y, centro_x = alto // 2, ancho // 2
    mascara = np.zeros((alto, ancho), dtype=np.float64)

    for y in range(alto):
        for x in range(ancho):
            distancia = np.sqrt((y - centro_y) ** 2 + (x - centro_x) ** 2)
            if distancia <= radio:
                mascara[y, x] = 1.0

    # Aplicar mascara al espectro (equivale a convolución en dominio espacial)
    espectro_filtrado = espectro_centrado * mascara

    # Transformada inversa
    espectro_deshift = np.fft.ifftshift(espectro_filtrado)
    imagen_filtrada = np.fft.ifft2(espectro_deshift)

    # Tomamos la magnitud (parte real) y normalizamos
    resultado = np.abs(imagen_filtrada)
    resultado = np.clip(resultado, 0, 255).astype(np.uint8)

    return resultado


def obtener_espectro_magnitud(imagen_gris):
    """
    Calcula el espectro de magnitud (log) para visualizacion.
    Se usa escala logaritmica porque los valores del espectro varian mucho.
    """
    espectro = np.fft.fft2(imagen_gris.astype(np.float64))
    espectro_centrado = np.fft.fftshift(espectro)
    magnitud = np.abs(espectro_centrado)

    # Log para comprimir el rango dinámico y que se vea bien
    magnitud_log = np.log1p(magnitud)

    # Normalizar a [0, 255]
    if magnitud_log.max() > 0:
        magnitud_log = (magnitud_log / magnitud_log.max()) * 255

    return magnitud_log.astype(np.uint8)


# --- Binarización ---

def binarizar(imagen_gris, umbral=None):
    """
    Binariza la imagen. Si no se pasa umbral, usa la media
    de los valores de la imagen como threshold.
    """
    if umbral is None:
        umbral = int(np.mean(imagen_gris))

    return np.where(imagen_gris >= umbral, 255, 0).astype(np.uint8)


def calcular_media(imagen_gris):
    """Calcula el valor medio de los pixeles."""
    return float(np.mean(imagen_gris))
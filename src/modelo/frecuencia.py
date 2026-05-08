"""Filtros en dominio de la frecuencia."""

import math

import numpy as np

from .base import convertir_a_grises
from .utils import (
    aplicar_por_canal,
    matriz_absoluta_manual,
    normalizar_matriz_uint8,
    parte_real_manual,
)

def crear_filtro_gaussiano(altura, ancho, d0):
    """
    Crea una máscara gaussiana pasabajas en frecuencia.
    Ejemplo corto: lo cercano al centro pasa casi completo y lo lejano se  apaga poco a poco, sin cortes bruscos.
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


def crear_filtro_ideal(altura, ancho, d0):
    """
    Máscara pasabajas ideal: deja pasar solo lo cercano al centro.
    Ejemplo corto: con D0=30, una frecuencia a distancia 20 pasa y a 40 se anula.
    """
    if d0 <= 0:
        d0 = 1.0

    cy = altura // 2
    cx = ancho // 2
    filtro = np.zeros((altura, ancho), dtype=np.float64)

    for fila in range(altura):
        for columna in range(ancho):
            distancia = math.sqrt((fila - cy) ** 2 + (columna - cx) ** 2)
            filtro[fila, columna] = 1.0 if distancia <= d0 else 0.0

    return filtro


def crear_filtro_butterworth(altura, ancho, d0, orden=2):
    """
    Máscara pasabajas Butterworth, más suave que la ideal.
    Ejemplo corto: cerca de D0 no salta de 1 a 0, baja gradualmente.
    """
    if d0 <= 0:
        d0 = 1.0

    cy = altura // 2
    cx = ancho // 2
    filtro = np.zeros((altura, ancho), dtype=np.float64)

    for fila in range(altura):
        for columna in range(ancho):
            distancia = math.sqrt((fila - cy) ** 2 + (columna - cx) ** 2)
            filtro[fila, columna] = 1.0 / (1.0 + (distancia / d0) ** (2 * orden))

    return filtro


def crear_mascara_frecuencia(altura, ancho, d0, tipo="Gaussiano", modo="pasa-bajas", factor=2.0):
    """
    Construye pasabajas, pasa-altas o high-boost desde la misma base.
    Ejemplo corto: si H deja pasar lo suave, entonces 1-H resalta cambios finos.
    """
    if factor < 1.0:
        factor = 1.0

    if tipo == "Ideal":
        pasabajas = crear_filtro_ideal(altura, ancho, d0)
    elif tipo == "Butterworth":
        pasabajas = crear_filtro_butterworth(altura, ancho, d0)
    else:
        pasabajas = crear_filtro_gaussiano(altura, ancho, d0)

    if modo == "pasa-bajas":
        return pasabajas

    mascara = np.zeros((altura, ancho), dtype=np.float64)
    for fila in range(altura):
        for columna in range(ancho):
            if modo == "high-boost":
                mascara[fila, columna] = factor - pasabajas[fila, columna]
            else:
                mascara[fila, columna] = 1.0 - pasabajas[fila, columna]

    return mascara


def aplicar_filtro_frecuencia(espectro, filtro):
    """Multiplica cada punto del espectro por el peso que le toca en la máscara."""
    alto, ancho = espectro.shape
    resultado = np.zeros_like(espectro)

    for fila in range(alto):
        for columna in range(ancho):
            resultado[fila, columna] = espectro[fila, columna] * filtro[fila, columna]

    return resultado


def espectro_log(espectro_complejo):
    """
    Convierte el espectro a una imagen visible con escala logarítmica.
    Ejemplo corto: el centro no tapa los detalles pequeños de las esquinas.
    """
    alto, ancho = espectro_complejo.shape
    magnitud = np.zeros((alto, ancho), dtype=np.float64)
    for fila in range(alto):
        for columna in range(ancho):
            v = espectro_complejo[fila, columna]
            mod = math.sqrt(v.real ** 2 + v.imag ** 2)
            magnitud[fila, columna] = math.log1p(mod)
    return normalizar_matriz_uint8(magnitud)


def filtrar_frecuencia_matriz(imagen, d0, tipo="Gaussiano", modo="pasa-bajas", factor=2.0):
    """
    Aplica una máscara de frecuencia y reconstruye la imagen filtrada.
    Ejemplo corto: pasabajas suaviza textura; pasa-altas deja principalmente bordes.
    """
    alto, ancho = imagen.shape
    espectro = np.fft.fft2(imagen.astype(np.float64))
    espectro_centrado = np.fft.fftshift(espectro)
    mascara = crear_mascara_frecuencia(alto, ancho, d0, tipo, modo, factor)
    espectro_filtrado = aplicar_filtro_frecuencia(espectro_centrado, mascara)
    reconstruida = np.fft.ifft2(np.fft.ifftshift(espectro_filtrado))
    respuesta_real = parte_real_manual(reconstruida)
    if modo == "pasa-altas":
        # Para bordes se conserva la fuerza del cambio; el signo solo indica sentido.
        resultado = normalizar_matriz_uint8(matriz_absoluta_manual(respuesta_real))
    else:
        resultado = normalizar_matriz_uint8(respuesta_real)
    return resultado, espectro_centrado, espectro_filtrado, mascara


def fourier_filtrar_canal(imagen, d0, tipo="Gaussiano", modo="pasa-bajas", factor=2.0):
    """Aplica filtrado en frecuencia a un canal."""
    resultado, _, _, _ = filtrar_frecuencia_matriz(imagen, d0, tipo, modo, factor)
    return resultado


def filtro_frecuencia_gaussiano(imagen, d0, tipo="Gaussiano"):
    """Aplica suavizado frecuencial a una imagen en gris o RGB."""
    print(f"[modelo] Aplicando suavizado {tipo} en frecuencia con D0={d0}...")
    return aplicar_por_canal(imagen, lambda canal: fourier_filtrar_canal(canal, d0, tipo))


def diagnostico_frecuencia(imagen, d0, tipo="Gaussiano"):
    """Genera imágenes de apoyo para la pestaña de frecuencia."""
    print("[modelo] Armando diagnostico de frecuencia...")
    if imagen.ndim == 3:
        base = convertir_a_grises(imagen)
    else:
        base = imagen

    resultado, espectro_original, espectro_filtrado, mascara = filtrar_frecuencia_matriz(
        base, d0, tipo
    )
    return {
        "base_gris": base,
        "resultado_gris": resultado,
        "espectro_original": espectro_log(espectro_original),
        "mascara": normalizar_matriz_uint8(mascara),
        "espectro_filtrado": espectro_log(espectro_filtrado),
    }


def filtro_frecuencia_pasaaltas(imagen, d0, tipo="Gaussiano", factor=2.0):
    """
    Filtro pasa-altas en frecuencia.
    Ejemplo corto: al quitar las bajas frecuencias, queda lo que cambia rápido:
    bordes, textura y ruido fino.
    """
    modo = "high-boost" if tipo == "High-Boost" else "pasa-altas"
    tipo_mascara = "Gaussiano" if tipo == "High-Boost" else tipo
    print(f"[modelo] Aplicando {tipo} en frecuencia con D0={d0}...")

    def _canal_pasaaltas(canal):
        return fourier_filtrar_canal(canal, d0, tipo_mascara, modo, factor)

    return aplicar_por_canal(imagen, _canal_pasaaltas)


def diagnostico_pasaaltas(imagen, d0, tipo="Gaussiano", factor=2.0):
    """Genera imágenes de apoyo para revisar el pasa-altas en frecuencia."""
    print("[modelo] Armando diagnostico pasa-altas...")
    if imagen.ndim == 3:
        base = convertir_a_grises(imagen)
    else:
        base = imagen

    modo = "high-boost" if tipo == "High-Boost" else "pasa-altas"
    tipo_mascara = "Gaussiano" if tipo == "High-Boost" else tipo
    resultado, espectro_centrado, espectro_filtrado, mascara_pa = filtrar_frecuencia_matriz(
        base, d0, tipo_mascara, modo, factor
    )

    return {
        "espectro_original_pa": espectro_log(espectro_centrado),
        "mascara_pasaaltas":    normalizar_matriz_uint8(mascara_pa),
        "espectro_filtrado_pa": espectro_log(espectro_filtrado),
        "pasaaltas_resultado":  resultado,
    }

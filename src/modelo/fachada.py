"""Punto de reunión para llamar los algoritmos desde la interfaz."""

from .base import (
    cargar_imagen,
    convertir_a_grises,
    normalizar_histograma_grises,
    calcular_histograma_grises,
    calcular_tamano_mascara_maximo,
    agregar_ruido_sal_pimienta,
)
from .suavizado import filtro_media, filtro_mediana, filtro_moda
from .frecuencia import (
    filtro_frecuencia_gaussiano,
    diagnostico_frecuencia,
    filtro_frecuencia_pasaaltas,
    diagnostico_pasaaltas,
)
from .gradientes import (
    diferencia_absoluta_manual,
    filtro_roberts,
    filtro_prewitt,
    filtro_sobel,
    filtro_laplaciano,
    filtro_pasa_alto,
    filtro_high_boost,
    gradientes_bordes,
    diagnostico_gradiente,
)
from .regiones import (
    binarizar_imagen,
    etiquetar_regiones_bfs,
    limpiar_mascara_regiones,
    filtrar_regiones_utiles,
    dibujar_bounding_boxes,
    dibujar_bounding_boxes_sobre_imagen,
    extraer_recortes_normalizados,
    preparar_entrada_red_neuronal,
    componer_tira_recortes,
)


class ModeloImagen:
    """Agrupa las operaciones principales sin mezclar la UI con los algoritmos."""

    def cargar_imagen(self, ruta):
        return cargar_imagen(ruta)

    def convertir_a_grises(self, imagen_rgb):
        return convertir_a_grises(imagen_rgb)

    def normalizar_histograma_grises(self, imagen_gris):
        return normalizar_histograma_grises(imagen_gris)

    def calcular_histograma_grises(self, imagen_gris):
        return calcular_histograma_grises(imagen_gris)

    def calcular_tamano_mascara_maximo(self, imagen):
        return calcular_tamano_mascara_maximo(imagen)

    def agregar_ruido_sal_pimienta(self, imagen, intensidad):
        return agregar_ruido_sal_pimienta(imagen, intensidad)

    def filtro_media(self, imagen, tamano_mascara):
        return filtro_media(imagen, tamano_mascara)

    def filtro_mediana(self, imagen, tamano_mascara):
        return filtro_mediana(imagen, tamano_mascara)

    def filtro_moda(self, imagen, tamano_mascara):
        return filtro_moda(imagen, tamano_mascara)

    def filtro_frecuencia_gaussiano(self, imagen, d0, tipo="Gaussiano"):
        return filtro_frecuencia_gaussiano(imagen, d0, tipo)

    def diagnostico_frecuencia(self, imagen, d0, tipo="Gaussiano"):
        return diagnostico_frecuencia(imagen, d0, tipo)

    def diferencia_absoluta_manual(self, imagen_a, imagen_b):
        return diferencia_absoluta_manual(imagen_a, imagen_b)

    def filtro_roberts(self, imagen):
        return filtro_roberts(imagen)

    def filtro_prewitt(self, imagen):
        return filtro_prewitt(imagen)

    def filtro_sobel(self, imagen):
        return filtro_sobel(imagen)

    def filtro_laplaciano(self, imagen):
        return filtro_laplaciano(imagen)

    def filtro_pasa_alto(self, imagen):
        return filtro_pasa_alto(imagen)

    def filtro_high_boost(self, imagen, factor):
        return filtro_high_boost(imagen, factor)

    def gradientes_bordes(self, imagen):
        return gradientes_bordes(imagen)

    def diagnostico_gradiente(self, imagen, operador):
        return diagnostico_gradiente(imagen, operador)

    def filtro_frecuencia_pasaaltas(self, imagen, d0, tipo="Gaussiano", factor=2.0):
        return filtro_frecuencia_pasaaltas(imagen, d0, tipo, factor)

    def diagnostico_pasaaltas(self, imagen, d0, tipo="Gaussiano", factor=2.0):
        return diagnostico_pasaaltas(imagen, d0, tipo, factor)

    def binarizar_imagen(self, imagen_gris, umbral):
        return binarizar_imagen(imagen_gris, umbral)

    def etiquetar_regiones_bfs(self, imagen_binaria, min_area=50, max_area=None):
        return etiquetar_regiones_bfs(imagen_binaria, min_area, max_area)

    def limpiar_mascara_regiones(self, imagen_binaria):
        return limpiar_mascara_regiones(imagen_binaria)

    def filtrar_regiones_utiles(self, regiones, alto, ancho, max_area=None):
        return filtrar_regiones_utiles(regiones, alto, ancho, max_area)

    def dibujar_bounding_boxes(self, imagen_binaria, regiones):
        return dibujar_bounding_boxes(imagen_binaria, regiones)

    def dibujar_bounding_boxes_sobre_imagen(self, imagen_base, regiones):
        return dibujar_bounding_boxes_sobre_imagen(imagen_base, regiones)

    def extraer_recortes_normalizados(self, imagen, regiones, tamano=64):
        return extraer_recortes_normalizados(imagen, regiones, tamano)

    def preparar_entrada_red_neuronal(self, recortes):
        return preparar_entrada_red_neuronal(recortes)

    def componer_tira_recortes(self, imagen, regiones, tamano=64, separacion=6):
        return componer_tira_recortes(imagen, regiones, tamano, separacion)


modelo_imagen = ModeloImagen()

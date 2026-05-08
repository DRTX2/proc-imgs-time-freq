from dataclasses import dataclass
from typing import Any

import modelo


@dataclass(frozen=True)
class ParametrosProcesamiento:
    """Agrupa los controles que la UI envía al pipeline."""

    img_rgb: Any
    ruido: float
    mascara: int
    d0_suavizado: int
    d0_acentuado: int
    dominio_suavizado: str
    tipo_suavizado: str
    dominio_acentuado: str
    tipo_acentuado: str
    tipo_gradiente: str
    umbral: int = 128
    min_area: int = 50
    max_area: int = 0


@dataclass
class ResultadoProcesamiento:
    """Contiene todas las salidas generadas por el procesamiento completo."""

    original: Any
    gris: Any
    hist_gris: list[int]
    normalizada: Any
    hist_normalizada: list[int]
    imagen_ruido: Any
    suavizada: Any
    mapa_cambio: Any
    diagnostico_suavizado: dict[str, Any]
    acentuada: Any
    diagnostico_acentuado: dict[str, Any]
    binaria_acentuada: Any
    gradiente_final: Any
    gradiente_binario: Any
    componentes_gradiente: dict[str, Any]
    frecuencia: Any
    espectro_original: Any
    mascara_frecuencia: Any
    espectro_filtrado: Any
    espectro_original_pa: Any
    mascara_pasaaltas: Any
    espectro_filtrado_pa: Any
    pasaaltas_resultado: Any
    binaria: Any
    bboxes: Any
    recortes_tira: Any
    n_regiones: int
    regiones: list[dict[str, Any]]
    umbral: int
    min_area: int
    max_area: int
    dominio_suavizado: str
    tipo_suavizado: str
    dominio_acentuado: str
    tipo_acentuado: str
    tipo_gradiente: str
    mascara: int
    d0_suavizado: int
    d0_acentuado: int
    ruido_porcentaje: int

    def a_diccionario(self):
        """Mantiene compatibilidad con la capa de visualización actual."""
        return {
            "original": self.original,
            "gris": self.gris,
            "hist_gris": self.hist_gris,
            "normalizada": self.normalizada,
            "hist_normalizada": self.hist_normalizada,
            "ruido": self.imagen_ruido,
            "suavizada": self.suavizada,
            "mapa_cambio": self.mapa_cambio,
            "diagnostico_suavizado": self.diagnostico_suavizado,
            "acentuada": self.acentuada,
            "diagnostico_acentuado": self.diagnostico_acentuado,
            "binaria_acentuada": self.binaria_acentuada,
            "gradiente_final": self.gradiente_final,
            "gradiente_binario": self.gradiente_binario,
            "componentes_gradiente": self.componentes_gradiente,
            "frecuencia": self.frecuencia,
            "espectro_original": self.espectro_original,
            "mascara_frecuencia": self.mascara_frecuencia,
            "espectro_filtrado": self.espectro_filtrado,
            "espectro_original_pa": self.espectro_original_pa,
            "mascara_pasaaltas": self.mascara_pasaaltas,
            "espectro_filtrado_pa": self.espectro_filtrado_pa,
            "pasaaltas_resultado": self.pasaaltas_resultado,
            "binaria": self.binaria,
            "bboxes": self.bboxes,
            "recortes_tira": self.recortes_tira,
            "n_regiones": self.n_regiones,
            "regiones": self.regiones,
            "umbral": self.umbral,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "dominio_suavizado": self.dominio_suavizado,
            "tipo_suavizado": self.tipo_suavizado,
            "dominio_acentuado": self.dominio_acentuado,
            "tipo_acentuado": self.tipo_acentuado,
            "tipo_gradiente": self.tipo_gradiente,
            "mascara": self.mascara,
            "d0_suavizado": self.d0_suavizado,
            "d0_acentuado": self.d0_acentuado,
            "ruido_porcentaje": self.ruido_porcentaje,
        }


class ProcesadorImagen:
    """Orquesta el pipeline completo usando las funciones de `modelo.py`."""

    def __init__(self, modelo_algoritmos=None):
        self.modelo = modelo_algoritmos or modelo.modelo_imagen

    def ejecutar(self, parametros: ParametrosProcesamiento):
        print("[pipeline] Iniciando procesamiento completo...")

        img_gris = self.modelo.convertir_a_grises(parametros.img_rgb)
        hist_gris = self.modelo.calcular_histograma_grises(img_gris)
        print("[pipeline] Grises listos.")

        img_normalizada = self.modelo.normalizar_histograma_grises(img_gris)
        hist_normalizada = self.modelo.calcular_histograma_grises(img_normalizada)
        print("[pipeline] Normalizacion lista.")

        img_ruido = self.modelo.agregar_ruido_sal_pimienta(img_normalizada, parametros.ruido)
        print("[pipeline] Ruido listo.")

        imagen_suavizada = self._aplicar_filtro_suavizado(
            img_ruido,
            parametros.dominio_suavizado,
            parametros.tipo_suavizado,
            parametros.mascara,
            parametros.d0_suavizado,
        )
        diagnostico_suavizado = self._diagnostico_suavizado(
            img_ruido,
            parametros.dominio_suavizado,
            parametros.d0_suavizado,
        )
        mapa_cambio = self.modelo.diferencia_absoluta_manual(img_ruido, imagen_suavizada)
        print(f"[pipeline] Suavizado listo: {parametros.dominio_suavizado} / {parametros.tipo_suavizado}.")

        imagen_acentuada = self._aplicar_filtro_acentuado(
            imagen_suavizada,
            parametros.dominio_acentuado,
            parametros.tipo_acentuado,
            parametros.d0_acentuado,
        )
        diagnostico_acentuado = self._diagnostico_acentuado(
            imagen_suavizada,
            parametros.dominio_acentuado,
            parametros.d0_acentuado,
        )
        print(f"[pipeline] Acentuado listo: {parametros.dominio_acentuado} / {parametros.tipo_acentuado}.")

        img_binaria_acentuada = self.modelo.binarizar_imagen(imagen_acentuada, parametros.umbral)
        print("[pipeline] Binarización post-acentuado lista.")

        gradiente_final = self._aplicar_gradiente(
            img_binaria_acentuada,
            parametros.tipo_gradiente,
        )
        img_gradiente_binario = self.modelo.binarizar_imagen(gradiente_final, 1)
        componentes_gradiente = self.modelo.diagnostico_gradiente(
            img_binaria_acentuada,
            parametros.tipo_gradiente,
        )
        print(f"[pipeline] Gradiente final listo: {parametros.tipo_gradiente}.")

        max_area = parametros.max_area if parametros.max_area > 0 else None
        regiones = self.modelo.etiquetar_regiones_bfs(
            img_gradiente_binario,
            parametros.min_area,
            max_area,
        )
        img_bboxes = self.modelo.dibujar_bounding_boxes_sobre_imagen(parametros.img_rgb, regiones)
        recortes_tira = self.modelo.componer_tira_recortes(img_gradiente_binario, regiones)
        print(f"[pipeline] Binarización lista. Regiones: {len(regiones)}")

        print("[pipeline] Todo el proceso termino bien.")
        return ResultadoProcesamiento(
            original=parametros.img_rgb,
            gris=img_gris,
            hist_gris=hist_gris,
            normalizada=img_normalizada,
            hist_normalizada=hist_normalizada,
            imagen_ruido=img_ruido,
            suavizada=imagen_suavizada,
            mapa_cambio=mapa_cambio,
            diagnostico_suavizado=diagnostico_suavizado,
            acentuada=imagen_acentuada,
            diagnostico_acentuado=diagnostico_acentuado,
            binaria_acentuada=img_binaria_acentuada,
            gradiente_final=gradiente_final,
            gradiente_binario=img_gradiente_binario,
            componentes_gradiente=componentes_gradiente,
            frecuencia=diagnostico_suavizado["salida"],
            espectro_original=diagnostico_suavizado["espectro_original"],
            mascara_frecuencia=diagnostico_suavizado["mascara"],
            espectro_filtrado=diagnostico_suavizado["espectro_filtrado"],
            espectro_original_pa=diagnostico_acentuado["espectro_original"],
            mascara_pasaaltas=diagnostico_acentuado["mascara"],
            espectro_filtrado_pa=diagnostico_acentuado["espectro_filtrado"],
            pasaaltas_resultado=diagnostico_acentuado["salida"],
            binaria=img_gradiente_binario,
            bboxes=img_bboxes,
            recortes_tira=recortes_tira,
            n_regiones=len(regiones),
            regiones=regiones,
            umbral=parametros.umbral,
            min_area=parametros.min_area,
            max_area=parametros.max_area,
            dominio_suavizado=parametros.dominio_suavizado,
            tipo_suavizado=parametros.tipo_suavizado,
            dominio_acentuado=parametros.dominio_acentuado,
            tipo_acentuado=parametros.tipo_acentuado,
            tipo_gradiente=parametros.tipo_gradiente,
            mascara=parametros.mascara,
            d0_suavizado=parametros.d0_suavizado,
            d0_acentuado=parametros.d0_acentuado,
            ruido_porcentaje=int(round(parametros.ruido * 100)),
        )

    def _aplicar_filtro_suavizado(self, imagen, dominio, tipo_filtro, tam_mascara, d0):
        if dominio == "Frecuencial":
            return self.modelo.filtro_frecuencia_gaussiano(imagen, d0)
        if tipo_filtro == "Media":
            return self.modelo.filtro_media(imagen, tam_mascara)
        if tipo_filtro == "Mediana":
            return self.modelo.filtro_mediana(imagen, tam_mascara)
        return self.modelo.filtro_moda(imagen, tam_mascara)

    def _aplicar_filtro_acentuado(self, imagen, dominio, tipo_filtro, d0):
        if dominio == "Frecuencial":
            return self.modelo.filtro_frecuencia_pasaaltas(imagen, d0)
        if tipo_filtro == "Roberts":
            return self.modelo.filtro_roberts(imagen)
        if tipo_filtro == "Prewitt":
            return self.modelo.filtro_prewitt(imagen)
        if tipo_filtro == "Sobel":
            return self.modelo.filtro_sobel(imagen)
        return self.modelo.filtro_laplaciano(imagen)

    def _diagnostico_suavizado(self, imagen, dominio, d0):
        if dominio == "Frecuencial":
            diagnostico = self.modelo.diagnostico_frecuencia(imagen, d0)
            return {
                "espectro_original": diagnostico["espectro_original"],
                "mascara": diagnostico["mascara"],
                "espectro_filtrado": diagnostico["espectro_filtrado"],
                "salida": diagnostico["resultado_gris"],
            }
        return {
            "espectro_original": None,
            "mascara": None,
            "espectro_filtrado": None,
            "salida": imagen,
        }

    def _diagnostico_acentuado(self, imagen, dominio, d0):
        if dominio == "Frecuencial":
            diagnostico = self.modelo.diagnostico_pasaaltas(imagen, d0)
            return {
                "espectro_original": diagnostico["espectro_original_pa"],
                "mascara": diagnostico["mascara_pasaaltas"],
                "espectro_filtrado": diagnostico["espectro_filtrado_pa"],
                "salida": diagnostico["pasaaltas_resultado"],
            }
        return {
            "espectro_original": None,
            "mascara": None,
            "espectro_filtrado": None,
            "salida": imagen,
        }

    def _aplicar_gradiente(self, imagen, tipo_filtro):
        gradientes = self.modelo.gradientes_bordes(imagen)
        if tipo_filtro == "Roberts":
            return gradientes["roberts_grad"]
        if tipo_filtro == "Prewitt":
            return gradientes["prewitt_grad"]
        if tipo_filtro == "Sobel":
            return gradientes["sobel_grad"]
        if tipo_filtro == "Kirsch":
            return gradientes["kirsch_grad"]
        return gradientes["laplaciano_grad"]

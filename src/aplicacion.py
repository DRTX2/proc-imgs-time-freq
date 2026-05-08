from dataclasses import dataclass
from typing import Any

from src import modelo


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
    tipo_frecuencia_suavizado: str
    dominio_acentuado: str
    tipo_acentuado: str
    tipo_frecuencia_acentuado: str
    factor_high_boost: float
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
    recortes_clasificador: list[dict[str, Any]]
    entradas_red_neuronal: list[dict[str, Any]]
    n_regiones: int
    regiones: list[dict[str, Any]]
    umbral: int
    min_area: int
    max_area: int
    dominio_suavizado: str
    tipo_suavizado: str
    tipo_frecuencia_suavizado: str
    dominio_acentuado: str
    tipo_acentuado: str
    tipo_frecuencia_acentuado: str
    factor_high_boost: float
    tipo_gradiente: str
    mascara: int
    d0_suavizado: int
    d0_acentuado: int
    ruido_porcentaje: int

    def a_diccionario(self):
        """Entrega las salidas con nombres directos para pintarlas o exportarlas."""
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
            "recortes_clasificador": self.recortes_clasificador,
            "entradas_red_neuronal": self.entradas_red_neuronal,
            "n_regiones": self.n_regiones,
            "regiones": self.regiones,
            "umbral": self.umbral,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "dominio_suavizado": self.dominio_suavizado,
            "tipo_suavizado": self.tipo_suavizado,
            "tipo_frecuencia_suavizado": self.tipo_frecuencia_suavizado,
            "dominio_acentuado": self.dominio_acentuado,
            "tipo_acentuado": self.tipo_acentuado,
            "tipo_frecuencia_acentuado": self.tipo_frecuencia_acentuado,
            "factor_high_boost": self.factor_high_boost,
            "tipo_gradiente": self.tipo_gradiente,
            "mascara": self.mascara,
            "d0_suavizado": self.d0_suavizado,
            "d0_acentuado": self.d0_acentuado,
            "ruido_porcentaje": self.ruido_porcentaje,
        }


class ProcesadorImagen:
    """Orquesta el recorrido completo: preparación, filtros, bordes y regiones."""

    def __init__(self, modelo_algoritmos=None):
        self.modelo = modelo_algoritmos or modelo.modelo_imagen

    def ejecutar_hasta(self, parametros: ParametrosProcesamiento, etapa: str):
        """Procesa solo hasta la etapa pedida para que la UI no recalcule todo."""
        print(f"[pipeline] Procesando hasta: {etapa}...")

        img_gris = self.modelo.convertir_a_grises(parametros.img_rgb)
        hist_gris = self.modelo.calcular_histograma_grises(img_gris)
        img_ruido = self.modelo.agregar_ruido_sal_pimienta(img_gris, parametros.ruido)
        img_normalizada = self.modelo.normalizar_histograma_grises(img_ruido)
        hist_normalizada = self.modelo.calcular_histograma_grises(img_normalizada)

        datos = {
            "original": parametros.img_rgb,
            "gris": img_gris,
            "hist_gris": hist_gris,
            "ruido": img_ruido,
            "normalizada": img_normalizada,
            "hist_normalizada": hist_normalizada,
            "ruido_porcentaje": int(round(parametros.ruido * 100)),
            "umbral": parametros.umbral,
            "min_area": parametros.min_area,
            "max_area": parametros.max_area,
            "dominio_suavizado": parametros.dominio_suavizado,
            "tipo_suavizado": parametros.tipo_suavizado,
            "tipo_frecuencia_suavizado": parametros.tipo_frecuencia_suavizado,
            "dominio_acentuado": parametros.dominio_acentuado,
            "tipo_acentuado": parametros.tipo_acentuado,
            "tipo_frecuencia_acentuado": parametros.tipo_frecuencia_acentuado,
            "factor_high_boost": parametros.factor_high_boost,
            "tipo_gradiente": parametros.tipo_gradiente,
            "mascara": parametros.mascara,
            "d0_suavizado": parametros.d0_suavizado,
            "d0_acentuado": parametros.d0_acentuado,
        }
        if etapa == "preprocesamiento":
            return datos

        imagen_suavizada = self._aplicar_filtro_suavizado(
            img_normalizada,
            parametros.dominio_suavizado,
            parametros.tipo_suavizado,
            parametros.mascara,
            parametros.d0_suavizado,
            parametros.tipo_frecuencia_suavizado,
        )
        diagnostico_suavizado = self._diagnostico_suavizado(
            img_normalizada,
            parametros.dominio_suavizado,
            parametros.d0_suavizado,
            parametros.tipo_frecuencia_suavizado,
        )
        datos.update({
            "suavizada": imagen_suavizada,
            "mapa_cambio": self.modelo.diferencia_absoluta_manual(img_normalizada, imagen_suavizada),
            "diagnostico_suavizado": diagnostico_suavizado,
            "frecuencia": diagnostico_suavizado["salida"],
            "espectro_original": diagnostico_suavizado["espectro_original"],
            "mascara_frecuencia": diagnostico_suavizado["mascara"],
            "espectro_filtrado": diagnostico_suavizado["espectro_filtrado"],
        })
        if etapa == "suavizado":
            return datos

        imagen_acentuada = self._aplicar_filtro_acentuado(
            imagen_suavizada,
            parametros.dominio_acentuado,
            parametros.tipo_acentuado,
            parametros.d0_acentuado,
            parametros.tipo_frecuencia_acentuado,
            parametros.factor_high_boost,
        )
        diagnostico_acentuado = self._diagnostico_acentuado(
            imagen_suavizada,
            parametros.dominio_acentuado,
            parametros.d0_acentuado,
            parametros.tipo_frecuencia_acentuado,
            parametros.factor_high_boost,
        )
        datos.update({
            "acentuada": imagen_acentuada,
            "diagnostico_acentuado": diagnostico_acentuado,
            "espectro_original_pa": diagnostico_acentuado["espectro_original"],
            "mascara_pasaaltas": diagnostico_acentuado["mascara"],
            "espectro_filtrado_pa": diagnostico_acentuado["espectro_filtrado"],
            "pasaaltas_resultado": diagnostico_acentuado["salida"],
        })
        if etapa == "acentuado":
            return datos

        img_binaria_acentuada = self.modelo.binarizar_imagen(imagen_acentuada, parametros.umbral)
        gradiente_final = self._aplicar_gradiente(img_binaria_acentuada, parametros.tipo_gradiente)
        img_gradiente_binario = self.modelo.binarizar_imagen(gradiente_final, 1)
        componentes_gradiente = self.modelo.diagnostico_gradiente(
            img_binaria_acentuada,
            parametros.tipo_gradiente,
        )
        datos.update({
            "binaria_acentuada": img_binaria_acentuada,
            "gradiente_final": gradiente_final,
            "gradiente_binario": img_gradiente_binario,
            "componentes_gradiente": componentes_gradiente,
        })
        if etapa == "gradiente":
            return datos

        mascara_regiones = self.modelo.limpiar_mascara_regiones(img_gradiente_binario)
        max_area = parametros.max_area if parametros.max_area > 0 else None
        regiones_base = self.modelo.etiquetar_regiones_bfs(
            mascara_regiones,
            parametros.min_area,
            max_area,
        )
        alto_mascara, ancho_mascara = mascara_regiones.shape
        regiones = self.modelo.filtrar_regiones_utiles(
            regiones_base,
            alto_mascara,
            ancho_mascara,
            max_area,
        )
        recortes_clasificador = self.modelo.extraer_recortes_normalizados(
            mascara_regiones,
            regiones,
        )
        datos.update({
            "binaria": mascara_regiones,
            "bboxes": self.modelo.dibujar_bounding_boxes_sobre_imagen(parametros.img_rgb, regiones),
            "recortes_tira": self.modelo.componer_tira_recortes(mascara_regiones, regiones),
            "recortes_clasificador": recortes_clasificador,
            "entradas_red_neuronal": self.modelo.preparar_entrada_red_neuronal(recortes_clasificador),
            "n_regiones": len(regiones),
            "regiones": regiones,
        })
        return datos

    def ejecutar(self, parametros: ParametrosProcesamiento):
        print("[pipeline] Iniciando procesamiento completo...")
        datos = self.ejecutar_hasta(parametros, "regiones")

        print("[pipeline] Todo el proceso termino bien.")
        return ResultadoProcesamiento(
            original=datos["original"],
            gris=datos["gris"],
            hist_gris=datos["hist_gris"],
            normalizada=datos["normalizada"],
            hist_normalizada=datos["hist_normalizada"],
            imagen_ruido=datos["ruido"],
            suavizada=datos["suavizada"],
            mapa_cambio=datos["mapa_cambio"],
            diagnostico_suavizado=datos["diagnostico_suavizado"],
            acentuada=datos["acentuada"],
            diagnostico_acentuado=datos["diagnostico_acentuado"],
            binaria_acentuada=datos["binaria_acentuada"],
            gradiente_final=datos["gradiente_final"],
            gradiente_binario=datos["gradiente_binario"],
            componentes_gradiente=datos["componentes_gradiente"],
            frecuencia=datos["frecuencia"],
            espectro_original=datos["espectro_original"],
            mascara_frecuencia=datos["mascara_frecuencia"],
            espectro_filtrado=datos["espectro_filtrado"],
            espectro_original_pa=datos["espectro_original_pa"],
            mascara_pasaaltas=datos["mascara_pasaaltas"],
            espectro_filtrado_pa=datos["espectro_filtrado_pa"],
            pasaaltas_resultado=datos["pasaaltas_resultado"],
            binaria=datos["binaria"],
            bboxes=datos["bboxes"],
            recortes_tira=datos["recortes_tira"],
            recortes_clasificador=datos["recortes_clasificador"],
            entradas_red_neuronal=datos["entradas_red_neuronal"],
            n_regiones=datos["n_regiones"],
            regiones=datos["regiones"],
            umbral=datos["umbral"],
            min_area=datos["min_area"],
            max_area=datos["max_area"],
            dominio_suavizado=datos["dominio_suavizado"],
            tipo_suavizado=datos["tipo_suavizado"],
            tipo_frecuencia_suavizado=datos["tipo_frecuencia_suavizado"],
            dominio_acentuado=datos["dominio_acentuado"],
            tipo_acentuado=datos["tipo_acentuado"],
            tipo_frecuencia_acentuado=datos["tipo_frecuencia_acentuado"],
            factor_high_boost=datos["factor_high_boost"],
            tipo_gradiente=datos["tipo_gradiente"],
            mascara=datos["mascara"],
            d0_suavizado=datos["d0_suavizado"],
            d0_acentuado=datos["d0_acentuado"],
            ruido_porcentaje=datos["ruido_porcentaje"],
        )

    def _aplicar_filtro_suavizado(self, imagen, dominio, tipo_filtro, tam_mascara, d0, tipo_frecuencia):
        if dominio == "Frecuencial":
            return self.modelo.filtro_frecuencia_gaussiano(imagen, d0, tipo_frecuencia)
        if tipo_filtro == "Media":
            return self.modelo.filtro_media(imagen, tam_mascara)
        if tipo_filtro == "Mediana":
            return self.modelo.filtro_mediana(imagen, tam_mascara)
        return self.modelo.filtro_moda(imagen, tam_mascara)

    def _aplicar_filtro_acentuado(self, imagen, dominio, tipo_filtro, d0, tipo_frecuencia, factor):
        if dominio == "Frecuencial":
            return self.modelo.filtro_frecuencia_pasaaltas(imagen, d0, tipo_frecuencia, factor)
        if tipo_filtro == "Pasa-alto":
            return self.modelo.filtro_pasa_alto(imagen)
        if tipo_filtro == "High-Boost":
            return self.modelo.filtro_high_boost(imagen, factor)
        return self.modelo.filtro_laplaciano(imagen)

    def _diagnostico_suavizado(self, imagen, dominio, d0, tipo_frecuencia):
        if dominio == "Frecuencial":
            diagnostico = self.modelo.diagnostico_frecuencia(imagen, d0, tipo_frecuencia)
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

    def _diagnostico_acentuado(self, imagen, dominio, d0, tipo_frecuencia, factor):
        if dominio == "Frecuencial":
            diagnostico = self.modelo.diagnostico_pasaaltas(imagen, d0, tipo_frecuencia, factor)
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

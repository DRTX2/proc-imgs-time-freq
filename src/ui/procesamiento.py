"""Acciones de carga, limpieza y ejecución del pipeline."""

from pathlib import Path

from PySide6.QtWidgets import QFileDialog

from src import modelo
from src.aplicacion import ParametrosProcesamiento

from .exportacion import guardar_resultados_finales
from .widgets import WorkerProcesoCompleto, WorkerProcesoEtapa


class ProcesamientoMixin:
    def _resultado_exportable(self):
        if self.ultimo_resultado is not None:
            return self.ultimo_resultado
        if all(clave in self.datos_parciales for clave in ("gradiente_final", "binaria", "bboxes")):
            return self.datos_parciales
        return None

    def _actualizar_estado_guardado(self, procesando=False):
        if hasattr(self, "btn_guardar_resultados"):
            self.btn_guardar_resultados.setEnabled(
                self._resultado_exportable() is not None and not procesando
            )

    def _parametros_actuales(self):
        return ParametrosProcesamiento(
            img_rgb=self.img_rgb.copy(),
            ruido=self.sl_ruido.value() / 100.0,
            mascara=self._kernel_actual(),
            d0_suavizado=self.sl_d0_suavizado.value(),
            d0_acentuado=self.sl_d0_acentuado.value(),
            dominio_suavizado=self.combo_dominio_suavizado.currentText(),
            tipo_suavizado=self.combo_suavizado.currentText(),
            tipo_frecuencia_suavizado=self.combo_frecuencia_suavizado.currentText(),
            dominio_acentuado=self.combo_dominio_acentuado.currentText(),
            tipo_acentuado=self.combo_acentuado.currentText(),
            tipo_frecuencia_acentuado=self.combo_frecuencia_acentuado.currentText(),
            factor_high_boost=self.sl_factor_acentuado.value() / 10,
            tipo_gradiente=self.combo_gradiente.currentText(),
            umbral=self.sl_umbral.value(),
            min_area=self.spin_min_area.value(),
            max_area=self.spin_max_area.value(),
        )

    def _texto_suavizado(self, datos):
        if datos["dominio_suavizado"] == "Frecuencial":
            return f"{datos['tipo_frecuencia_suavizado']} D0={datos['d0_suavizado']}"
        return datos["tipo_suavizado"]

    def _texto_acentuado(self, datos):
        if datos["dominio_acentuado"] == "Frecuencial":
            texto = f"{datos['tipo_frecuencia_acentuado']} D0={datos['d0_acentuado']}"
        else:
            texto = datos["tipo_acentuado"]

        if (
            (datos["dominio_acentuado"] == "Espacial" and datos["tipo_acentuado"] == "High-Boost")
            or (datos["dominio_acentuado"] == "Frecuencial" and datos["tipo_frecuencia_acentuado"] == "High-Boost")
        ):
            texto = f"{texto} A={datos['factor_high_boost']:.1f}"
        return texto

    def _limpiar_desde(self, etapa):
        if etapa in ("preprocesamiento",):
            self.canvas_suavizado.actualizar()
            self.canvas_acentuado.actualizar()
            self.canvas_binarizacion.actualizar()
            self.canvas_gradiente.actualizar()
            self.canvas_componentes_gradiente.actualizar()
            self.canvas_regiones.actualizar()
            self.canvas_recortes.actualizar()
            self.lbl_regiones_info.setText("Aplica las siguientes etapas para actualizar regiones.")
        elif etapa == "suavizado":
            self.canvas_acentuado.actualizar()
            self.canvas_binarizacion.actualizar()
            self.canvas_gradiente.actualizar()
            self.canvas_componentes_gradiente.actualizar()
            self.canvas_regiones.actualizar()
            self.canvas_recortes.actualizar()
            self.lbl_regiones_info.setText("Aplica acentuado y regiones para continuar.")
        elif etapa == "acentuado":
            self.canvas_binarizacion.actualizar()
            self.canvas_gradiente.actualizar()
            self.canvas_componentes_gradiente.actualizar()
            self.canvas_regiones.actualizar()
            self.canvas_recortes.actualizar()
            self.lbl_regiones_info.setText("Aplica regiones para detectar objetos.")

    def _actualizar_vistas_parciales(self, etapa, datos):
        self.datos_parciales = datos
        self.canvas_preprocesamiento.actualizar(datos)

        if etapa == "preprocesamiento":
            self._limpiar_desde(etapa)
            self._actualizar_estado_guardado()
            return

        self.canvas_suavizado.actualizar_titulos(
            [
                f"Normalizada post-ruido ({datos['ruido_porcentaje']} %)",
                f"Suavizado {datos['dominio_suavizado']} / {self._texto_suavizado(datos)}",
            ]
        )
        self.canvas_suavizado.actualizar(datos)
        if etapa == "suavizado":
            self._limpiar_desde(etapa)
            self._actualizar_estado_guardado()
            return

        self.canvas_acentuado.actualizar_titulos(
            [
                "Base suavizada",
                f"Acentuado {datos['dominio_acentuado']} / {self._texto_acentuado(datos)}",
            ]
        )
        self.canvas_acentuado.actualizar(datos)
        if etapa == "acentuado":
            self._limpiar_desde(etapa)
            self._actualizar_estado_guardado()
            return

        self.canvas_binarizacion.actualizar_titulos(
            [
                f"Acentuado {datos['dominio_acentuado']}",
                f"Máscara binaria (umbral={datos['umbral']})",
            ]
        )
        self.canvas_binarizacion.actualizar(datos)
        self.canvas_gradiente.actualizar_titulos(
            [
                f"Binarizada (umbral={datos['umbral']})",
                f"Gradiente {datos['tipo_gradiente']}",
                "Bordes binarios sin filtrar",
            ]
        )
        self.canvas_gradiente.actualizar(datos)
        componentes = datos["componentes_gradiente"]
        self.canvas_componentes_gradiente.actualizar_titulos(componentes["titulos"])
        self.canvas_componentes_gradiente.actualizar(
            {
                "comp_0": componentes["imagenes"][0],
                "comp_1": componentes["imagenes"][1],
                "comp_2": componentes["imagenes"][2],
            }
        )

        n = datos["n_regiones"]
        top3 = datos["regiones"][:3]
        resumen_top = "  |  ".join(
            f"R{r['label']}: {r['area']} px² / per={r['perimetro']}" for r in top3
        )
        texto_max = datos["max_area"] if datos["max_area"] > 0 else "sin límite"
        self.lbl_regiones_info.setText(
            f"Umbral: {datos['umbral']}  |  Área mín: {datos['min_area']} px²  |  Área máx: {texto_max}  |  "
            f"Regiones encontradas: {n}    Top-3 por área → {resumen_top if top3 else 'ninguna'}"
        )
        self.canvas_regiones.actualizar_titulos(
            [
                f"Máscara filtrada ({datos['tipo_gradiente']})",
                f"Bounding boxes ({n} regiones)",
            ]
        )
        self.canvas_regiones.actualizar(datos)
        self.canvas_recortes.actualizar_titulos(
            [
                f"Submatrices normalizadas para clasificación ({n} recortes)",
            ]
        )
        self.canvas_recortes.actualizar(datos)
        self._actualizar_estado_guardado()

    def _mostrar_placeholders(self):
        self.canvas_preprocesamiento.actualizar()
        self.canvas_suavizado.actualizar()
        self.canvas_acentuado.actualizar()
        self.canvas_gradiente.actualizar()
        self.canvas_componentes_gradiente.actualizar()
        self.canvas_binarizacion.actualizar()
        self.canvas_regiones.actualizar()
        self.canvas_recortes.actualizar()
        self.lbl_regiones_info.setText("Sin datos.")

    def _cargar_imagen(self):
        if self.worker and self.worker.isRunning():
            return

        ruta, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen",
            str(Path.cwd()),
            "Imagenes (*.png *.jpg *.jpeg *.bmp)",
        )

        if not ruta:
            return

        try:
            self.img_rgb = modelo.cargar_imagen(ruta)
            self.ultimo_resultado = None
            self.datos_parciales = {}
            alto, ancho = self.img_rgb.shape[:2]
            print(f"[ui] Imagen cargada en la interfaz: {Path(ruta).name}")

            self.lbl_path.setText(Path(ruta).name)
            self.lbl_size.setText(f"{ancho} x {alto} px")

            self.max_mascara = modelo.calcular_tamano_mascara_maximo(self.img_rgb)
            self._actualizar_opciones_kernel(self.max_mascara)
            self._actualizar_opciones_d0(alto, ancho)
            self._actualizar_opciones_area(alto, ancho)
            self._on_dominio_suavizado(self.combo_dominio_suavizado.currentText())
            self._on_dominio_acentuado(self.combo_dominio_acentuado.currentText())
            self._set_procesando(False)
            self.lbl_estado.setText("Imagen lista. Ajusta parámetros y pulsa Aplicar.")
            self.canvas_preprocesamiento.actualizar({"original": self.img_rgb})
            self.canvas_suavizado.actualizar()
            self.canvas_acentuado.actualizar()
            self.canvas_gradiente.actualizar()
            self.canvas_componentes_gradiente.actualizar()
            self.canvas_binarizacion.actualizar()
            self.canvas_regiones.actualizar()
            self.canvas_recortes.actualizar()
            self.lbl_regiones_info.setText("Sin datos.")
            self._actualizar_estado_guardado()
        except Exception as exc:
            self.lbl_estado.setText(f"Error al cargar la imagen: {exc}")

    def _limpiar_todo(self):
        if self.worker and self.worker.isRunning():
            return

        print("[ui] Limpiando resultados y reiniciando controles...")
        self.img_rgb = None
        self.ultimo_resultado = None
        self.datos_parciales = {}
        self.lbl_path.setText("Ninguna imagen cargada")
        self.lbl_size.setText("")
        self.lbl_estado.setText("Carga una imagen para empezar.")
        self.sl_ruido.setValue(5)
        self.sl_mascara.setValue(3)
        self.sl_d0_suavizado.setValue(45)
        self.sl_d0_acentuado.setValue(45)
        self.sl_umbral.setValue(128)
        self.spin_min_area.setValue(50)
        self.spin_max_area.setValue(0)
        self.combo_dominio_suavizado.setCurrentText("Espacial")
        self.combo_suavizado.setCurrentText("Media")
        self.combo_frecuencia_suavizado.setCurrentText("Gaussiano")
        self.combo_dominio_acentuado.setCurrentText("Espacial")
        self.combo_acentuado.setCurrentText("Laplaciano")
        self.combo_frecuencia_acentuado.setCurrentText("Gaussiano")
        self.sl_factor_acentuado.setValue(20)
        self.combo_gradiente.setCurrentText("Sobel")
        self._on_dominio_suavizado(self.combo_dominio_suavizado.currentText())
        self._on_dominio_acentuado(self.combo_dominio_acentuado.currentText())

        self._set_procesando(False)
        self._mostrar_placeholders()
        self._actualizar_resumen_flujo()

    def _procesar_todo(self):
        if self.img_rgb is None:
            return

        print("[ui] Lanzando procesamiento desde la interfaz...")
        self._set_procesando(True)
        self.lbl_estado.setText("Aplicando el pipeline seleccionado...")
        parametros = self._parametros_actuales()

        self.worker = WorkerProcesoCompleto(parametros)
        self.worker.terminado.connect(self._on_proceso_listo)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _guardar_resultados(self):
        resultado = self._resultado_exportable()
        if resultado is None:
            self.lbl_estado.setText("Primero aplica regiones para generar resultados finales.")
            return

        carpeta = QFileDialog.getExistingDirectory(
            self,
            "Guardar resultados finales",
            str(Path.cwd()),
        )
        if not carpeta:
            return

        try:
            archivos = guardar_resultados_finales(resultado, carpeta)
            self.lbl_estado.setText(
                f"Resultados guardados: {len(archivos)} archivo(s) en {Path(carpeta).name}."
            )
            print(f"[ui] Resultados finales guardados en: {carpeta}")
        except Exception as exc:
            self.lbl_estado.setText(f"No se pudieron guardar los resultados: {exc}")

    def _procesar_etapa(self, etapa):
        if self.img_rgb is None:
            return

        nombres = {
            "preprocesamiento": "escala de grises",
            "suavizado": "suavizado",
            "acentuado": "acentuado",
            "regiones": "detección de regiones",
        }
        print(f"[ui] Aplicando etapa: {etapa}...")
        self.ultimo_resultado = None
        self._set_procesando(True)
        self.lbl_estado.setText(f"Aplicando {nombres.get(etapa, etapa)}...")

        self.worker = WorkerProcesoEtapa(self._parametros_actuales(), etapa)
        self.worker.terminado.connect(self._on_etapa_lista)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_etapa_lista(self, etapa, datos):
        self._actualizar_vistas_parciales(etapa, datos)
        self.lbl_estado.setText(f"Etapa actualizada: {etapa}.")
        self._set_procesando(False)
        self.worker = None
        print(f"[ui] Etapa lista: {etapa}.")

    def _on_proceso_listo(self, resultado):
        print("[ui] Actualizando paneles con los resultados...")
        self.ultimo_resultado = resultado
        datos = resultado.a_diccionario()
        self.datos_parciales = datos
        self.canvas_preprocesamiento.actualizar(datos)

        filtro_suavizado = resultado.tipo_suavizado
        if resultado.dominio_suavizado == "Frecuencial":
            filtro_suavizado = f"{resultado.tipo_frecuencia_suavizado} D0={resultado.d0_suavizado}"

        filtro_acentuado = resultado.tipo_acentuado
        if resultado.dominio_acentuado == "Frecuencial":
            filtro_acentuado = f"{resultado.tipo_frecuencia_acentuado} D0={resultado.d0_acentuado}"
        if (
            (resultado.dominio_acentuado == "Espacial" and resultado.tipo_acentuado == "High-Boost")
            or (resultado.dominio_acentuado == "Frecuencial" and resultado.tipo_frecuencia_acentuado == "High-Boost")
        ):
            filtro_acentuado = f"{filtro_acentuado} A={resultado.factor_high_boost:.1f}"

        self.canvas_suavizado.actualizar_titulos(
            [
                f"Normalizada post-ruido ({resultado.ruido_porcentaje} %)",
                f"Suavizado {resultado.dominio_suavizado} / {filtro_suavizado}",
            ]
        )
        self.canvas_suavizado.actualizar(datos)
        self.canvas_acentuado.actualizar_titulos(
            [
                "Base suavizada",
                f"Acentuado {resultado.dominio_acentuado} / {filtro_acentuado}",
            ]
        )
        self.canvas_acentuado.actualizar(datos)
        self.canvas_binarizacion.actualizar_titulos(
            [
                f"Acentuado {resultado.dominio_acentuado}",
                f"Máscara binaria (umbral={resultado.umbral})",
            ]
        )
        self.canvas_binarizacion.actualizar(datos)
        self.canvas_gradiente.actualizar_titulos(
            [
                f"Binarizada (umbral={resultado.umbral})",
                f"Gradiente {resultado.tipo_gradiente}",
                "Bordes binarios sin filtrar",
            ]
        )
        self.canvas_gradiente.actualizar(datos)
        componentes = resultado.componentes_gradiente
        self.canvas_componentes_gradiente.actualizar_titulos(componentes["titulos"])
        self.canvas_componentes_gradiente.actualizar(
            {
                "comp_0": componentes["imagenes"][0],
                "comp_1": componentes["imagenes"][1],
                "comp_2": componentes["imagenes"][2],
            }
        )

        n = resultado.n_regiones
        umbral_usado = resultado.umbral
        min_area_usado = resultado.min_area
        max_area_usado = resultado.max_area
        top3 = resultado.regiones[:3]
        resumen_top = "  |  ".join(
            f"R{r['label']}: {r['area']} px² / per={r['perimetro']}" for r in top3
        )
        texto_max = max_area_usado if max_area_usado > 0 else "sin límite"
        self.lbl_regiones_info.setText(
            f"Umbral: {umbral_usado}  |  Área mín: {min_area_usado} px²  |  Área máx: {texto_max}  |  "
            f"Regiones encontradas: {n}    Top-3 por área → {resumen_top if top3 else 'ninguna'}"
        )
        self.canvas_regiones.actualizar_titulos(
            [
                f"Máscara filtrada ({resultado.tipo_gradiente})",
                f"Bounding boxes ({n} regiones)",
            ]
        )
        self.canvas_regiones.actualizar(datos)
        self.canvas_recortes.actualizar_titulos(
            [
                f"Submatrices normalizadas para clasificación ({n} recortes)",
            ]
        )
        self.canvas_recortes.actualizar(datos)

        self.lbl_estado.setText(
            f"Completado. Suavizado: {resultado.dominio_suavizado} | "
            f"Acentuado: {resultado.dominio_acentuado} | "
            f"Gradiente: {resultado.tipo_gradiente} | "
            f"Regiones: {n}."
        )
        self.btn_procesar.setEnabled(True)
        self._set_procesando(False)
        self.worker = None
        print("[ui] Interfaz lista para otra prueba.")

    def _on_error(self, mensaje):
        print(f"[ui] Se mostro un error en pantalla: {mensaje}")
        self.lbl_estado.setText(f"Error durante el proceso: {mensaje}")
        self._set_procesando(False)
        self.worker = None

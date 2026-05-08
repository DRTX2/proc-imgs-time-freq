"""Controles laterales y estado visual de parámetros."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QVBoxLayout

from .estilos import estilo_slider
from .widgets import ComboSoloDropdown


class ControlesMixin:
    def _crear_tarjeta_control(self, titulo, ayuda=None):
        tarjeta = QFrame()
        tarjeta.setObjectName("ControlCard")
        layout = QVBoxLayout(tarjeta)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        lbl_titulo = QLabel(titulo)
        lbl_titulo.setObjectName("CardTitle")
        lbl_titulo.setWordWrap(True)
        layout.addWidget(lbl_titulo)

        if ayuda:
            lbl_ayuda = QLabel(ayuda)
            lbl_ayuda.setObjectName("CardHint")
            lbl_ayuda.setWordWrap(True)
            layout.addWidget(lbl_ayuda)

        return tarjeta, layout

    def _agregar_bloque_ruido(self, layout_principal):
        tarjeta, bloque = self._crear_tarjeta_control(
            "Ruido sal y pimienta"
        )

        self.lbl_ruido = QLabel("5 %")
        self.lbl_ruido.setObjectName("ValueBadge")
        bloque.addWidget(self.lbl_ruido, alignment=Qt.AlignLeft)

        self.sl_ruido = QSlider(Qt.Horizontal)
        self.sl_ruido.setRange(0, 100)
        self.sl_ruido.setValue(5)
        self.sl_ruido.setStyleSheet(estilo_slider("#56B4D3"))
        self.sl_ruido.valueChanged.connect(self._actualizar_lbl_ruido)
        self.sl_ruido.valueChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.sl_ruido)

        self.btn_aplicar_pre = QPushButton("Aplicar escala")
        self.btn_aplicar_pre.setObjectName("ActionButton")
        self.btn_aplicar_pre.setEnabled(False)
        self.btn_aplicar_pre.clicked.connect(lambda: self._procesar_etapa("preprocesamiento"))
        bloque.addWidget(self.btn_aplicar_pre)

        layout_principal.addWidget(tarjeta)

    def _agregar_bloque_suavizado(self, layout_principal):
        tarjeta, bloque = self._crear_tarjeta_control(
            "Suavizado",
            "Escoge dominio y filtro.",
        )

        self.combo_dominio_suavizado = ComboSoloDropdown()
        self.combo_dominio_suavizado.addItems(["Espacial", "Frecuencial"])
        self.combo_dominio_suavizado.currentTextChanged.connect(self._on_dominio_suavizado)
        self.combo_dominio_suavizado.currentTextChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.combo_dominio_suavizado)

        self.lbl_suavizado = QLabel("Suavizado")
        self.lbl_suavizado.setObjectName("CardHint")
        bloque.addWidget(self.lbl_suavizado)

        self.combo_suavizado = ComboSoloDropdown()
        self.combo_suavizado.addItems(["Media", "Mediana", "Moda"])
        self.combo_suavizado.currentTextChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.combo_suavizado)

        self.lbl_kernel = QLabel("Mascara 3x3")
        self.lbl_kernel.setObjectName("ValueBadge")
        bloque.addWidget(self.lbl_kernel, alignment=Qt.AlignLeft)

        self.sl_mascara = QSlider(Qt.Horizontal)
        self.sl_mascara.setRange(3, 25)
        self.sl_mascara.setSingleStep(2)
        self.sl_mascara.setPageStep(2)
        self.sl_mascara.setValue(3)
        self.sl_mascara.setTickPosition(QSlider.TicksBelow)
        self.sl_mascara.setTickInterval(2)
        self.sl_mascara.setStyleSheet(estilo_slider("#7A8CF0"))
        self.sl_mascara.valueChanged.connect(self._on_mascara)
        bloque.addWidget(self.sl_mascara)

        self.lbl_frecuencia_suavizado = QLabel("Tipo frecuencial")
        self.lbl_frecuencia_suavizado.setObjectName("CardHint")
        bloque.addWidget(self.lbl_frecuencia_suavizado)

        self.combo_frecuencia_suavizado = ComboSoloDropdown()
        self.combo_frecuencia_suavizado.addItems(["Ideal", "Gaussiano", "Butterworth"])
        self.combo_frecuencia_suavizado.setCurrentText("Gaussiano")
        self.combo_frecuencia_suavizado.currentTextChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.combo_frecuencia_suavizado)

        self.lbl_d0_suavizado = QLabel("D0 de suavizado")
        self.lbl_d0_suavizado.setObjectName("CardHint")
        bloque.addWidget(self.lbl_d0_suavizado)

        self.lbl_d0_suavizado_valor = QLabel("D0 45 px")
        self.lbl_d0_suavizado_valor.setObjectName("ValueBadge")
        bloque.addWidget(self.lbl_d0_suavizado_valor, alignment=Qt.AlignLeft)

        self.sl_d0_suavizado = QSlider(Qt.Horizontal)
        self.sl_d0_suavizado.setRange(1, 180)
        self.sl_d0_suavizado.setValue(45)
        self.sl_d0_suavizado.setStyleSheet(estilo_slider("#F2B15E"))
        self.sl_d0_suavizado.valueChanged.connect(self._actualizar_lbl_d0_suavizado)
        self.sl_d0_suavizado.valueChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.sl_d0_suavizado)

        self.btn_aplicar_suavizado = QPushButton("Aplicar suavizado")
        self.btn_aplicar_suavizado.setObjectName("ActionButton")
        self.btn_aplicar_suavizado.setEnabled(False)
        self.btn_aplicar_suavizado.clicked.connect(lambda: self._procesar_etapa("suavizado"))
        bloque.addWidget(self.btn_aplicar_suavizado)

        layout_principal.addWidget(tarjeta)

    def _agregar_bloque_acentuado(self, layout_principal):
        tarjeta, bloque = self._crear_tarjeta_control(
            "Acentuado",
            "Espacial o frecuencial.",
        )

        self.combo_dominio_acentuado = ComboSoloDropdown()
        self.combo_dominio_acentuado.addItems(["Espacial", "Frecuencial"])
        self.combo_dominio_acentuado.currentTextChanged.connect(self._on_dominio_acentuado)
        self.combo_dominio_acentuado.currentTextChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.combo_dominio_acentuado)

        self.lbl_acentuado = QLabel("Operador")
        self.lbl_acentuado.setObjectName("CardHint")
        bloque.addWidget(self.lbl_acentuado)

        self.combo_acentuado = ComboSoloDropdown()
        self.combo_acentuado.addItems(["Laplaciano", "Pasa-alto", "High-Boost"])
        self.combo_acentuado.currentTextChanged.connect(lambda _: self._actualizar_estado_factor_acentuado())
        self.combo_acentuado.currentTextChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.combo_acentuado)

        self.lbl_frecuencia_acentuado = QLabel("Tipo frecuencial")
        self.lbl_frecuencia_acentuado.setObjectName("CardHint")
        bloque.addWidget(self.lbl_frecuencia_acentuado)

        self.combo_frecuencia_acentuado = ComboSoloDropdown()
        self.combo_frecuencia_acentuado.addItems(["Ideal", "Gaussiano", "Butterworth", "High-Boost"])
        self.combo_frecuencia_acentuado.setCurrentText("Gaussiano")
        self.combo_frecuencia_acentuado.currentTextChanged.connect(lambda _: self._actualizar_estado_factor_acentuado())
        self.combo_frecuencia_acentuado.currentTextChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.combo_frecuencia_acentuado)

        self.lbl_d0_acentuado = QLabel("D0 de acentuado")
        self.lbl_d0_acentuado.setObjectName("CardHint")
        bloque.addWidget(self.lbl_d0_acentuado)

        self.lbl_d0_acentuado_valor = QLabel("D0 45 px")
        self.lbl_d0_acentuado_valor.setObjectName("ValueBadge")
        bloque.addWidget(self.lbl_d0_acentuado_valor, alignment=Qt.AlignLeft)

        self.sl_d0_acentuado = QSlider(Qt.Horizontal)
        self.sl_d0_acentuado.setRange(1, 180)
        self.sl_d0_acentuado.setValue(45)
        self.sl_d0_acentuado.setStyleSheet(estilo_slider("#F2B15E"))
        self.sl_d0_acentuado.valueChanged.connect(self._actualizar_lbl_d0_acentuado)
        self.sl_d0_acentuado.valueChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.sl_d0_acentuado)

        self.lbl_factor_acentuado = QLabel("Factor A")
        self.lbl_factor_acentuado.setObjectName("CardHint")
        bloque.addWidget(self.lbl_factor_acentuado)

        self.lbl_factor_acentuado_valor = QLabel("A 2.0")
        self.lbl_factor_acentuado_valor.setObjectName("ValueBadge")
        bloque.addWidget(self.lbl_factor_acentuado_valor, alignment=Qt.AlignLeft)

        self.sl_factor_acentuado = QSlider(Qt.Horizontal)
        self.sl_factor_acentuado.setRange(10, 50)
        self.sl_factor_acentuado.setValue(20)
        self.sl_factor_acentuado.setStyleSheet(estilo_slider("#56B4D3"))
        self.sl_factor_acentuado.valueChanged.connect(self._actualizar_lbl_factor_acentuado)
        self.sl_factor_acentuado.valueChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.sl_factor_acentuado)

        self.btn_aplicar_acentuado = QPushButton("Aplicar acentuado")
        self.btn_aplicar_acentuado.setObjectName("ActionButton")
        self.btn_aplicar_acentuado.setEnabled(False)
        self.btn_aplicar_acentuado.clicked.connect(lambda: self._procesar_etapa("acentuado"))
        bloque.addWidget(self.btn_aplicar_acentuado)

        layout_principal.addWidget(tarjeta)

    def _agregar_bloque_binarizacion(self, layout_principal):
        tarjeta, bloque = self._crear_tarjeta_control("Binarización y Regiones")

        self.lbl_umbral = QLabel("Umbral 128")
        self.lbl_umbral.setObjectName("ValueBadge")
        bloque.addWidget(self.lbl_umbral, alignment=Qt.AlignLeft)

        self.sl_umbral = QSlider(Qt.Horizontal)
        self.sl_umbral.setRange(0, 255)
        self.sl_umbral.setValue(128)
        self.sl_umbral.setStyleSheet(estilo_slider("#A78BFA"))
        self.sl_umbral.valueChanged.connect(lambda v: self.lbl_umbral.setText(f"Umbral {v}"))
        self.sl_umbral.valueChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.sl_umbral)

        lbl_gradiente = QLabel("Gradiente")
        lbl_gradiente.setObjectName("CardHint")
        bloque.addWidget(lbl_gradiente)

        self.combo_gradiente = ComboSoloDropdown()
        self.combo_gradiente.addItems(["Roberts", "Prewitt", "Sobel", "Kirsch", "Laplaciano"])
        self.combo_gradiente.currentTextChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.combo_gradiente)

        fila_area = QHBoxLayout()
        fila_area.setSpacing(8)

        lbl_area = QLabel("Área(px)")
        lbl_area.setObjectName("CardHint")
        fila_area.addWidget(lbl_area)

        lbl_desde = QLabel("De")
        lbl_desde.setObjectName("CardHint")
        fila_area.addWidget(lbl_desde)

        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(1, 999999)
        self.spin_min_area.setValue(50)
        self.spin_min_area.valueChanged.connect(lambda _: self._actualizar_resumen_flujo())
        fila_area.addWidget(self.spin_min_area)

        lbl_hasta = QLabel("a")
        lbl_hasta.setObjectName("CardHint")
        fila_area.addWidget(lbl_hasta)

        self.spin_max_area = QSpinBox()
        self.spin_max_area.setRange(0, 999999)
        self.spin_max_area.setValue(0)
        self.spin_max_area.setSpecialValueText("sin límite")
        self.spin_max_area.valueChanged.connect(lambda _: self._actualizar_resumen_flujo())
        fila_area.addWidget(self.spin_max_area)

        bloque.addLayout(fila_area)

        self.btn_aplicar_regiones = QPushButton("Aplicar regiones")
        self.btn_aplicar_regiones.setObjectName("ActionButton")
        self.btn_aplicar_regiones.setEnabled(False)
        self.btn_aplicar_regiones.clicked.connect(lambda: self._procesar_etapa("regiones"))
        bloque.addWidget(self.btn_aplicar_regiones)

        layout_principal.addWidget(tarjeta)

    def _actualizar_lbl_ruido(self, valor):
        self.lbl_ruido.setText(f"{valor} %")

    def _actualizar_lbl_d0_suavizado(self, valor):
        self.lbl_d0_suavizado_valor.setText(f"D0 {valor} px")

    def _actualizar_lbl_d0_acentuado(self, valor):
        self.lbl_d0_acentuado_valor.setText(f"D0 {valor} px")

    def _actualizar_lbl_factor_acentuado(self, valor):
        self.lbl_factor_acentuado_valor.setText(f"A {valor / 10:.1f}")

    def _actualizar_opciones_kernel(self, maximo):
        if maximo % 2 == 0:
            maximo -= 1
        maximo = max(3, min(maximo, 25))

        self.sl_mascara.blockSignals(True)
        self.sl_mascara.setRange(3, maximo)
        self.sl_mascara.setValue(3)
        self.sl_mascara.blockSignals(False)
        self._on_mascara(3)

    def _actualizar_opciones_d0(self, alto, ancho):
        maximo = max(10, min(alto, ancho) // 2)
        sugerido = max(10, min(maximo, maximo // 3))

        self.sl_d0_suavizado.blockSignals(True)
        self.sl_d0_suavizado.setRange(1, maximo)
        self.sl_d0_suavizado.setValue(sugerido)
        self.sl_d0_suavizado.blockSignals(False)
        self._actualizar_lbl_d0_suavizado(sugerido)

        self.sl_d0_acentuado.blockSignals(True)
        self.sl_d0_acentuado.setRange(1, maximo)
        self.sl_d0_acentuado.setValue(sugerido)
        self.sl_d0_acentuado.blockSignals(False)
        self._actualizar_lbl_d0_acentuado(sugerido)

    def _actualizar_opciones_area(self, alto, ancho):
        total = max(1, alto * ancho)
        sugerido_max = min(total, max(500, total // 2))
        self.spin_min_area.setMaximum(total)
        self.spin_max_area.setMaximum(total)
        if self.spin_min_area.value() > total:
            self.spin_min_area.setValue(min(50, total))
        if self.spin_max_area.value() > total:
            self.spin_max_area.setValue(sugerido_max)
        self._actualizar_resumen_flujo()

    def _kernel_actual(self):
        valor = self.sl_mascara.value()
        if valor % 2 == 0:
            valor += 1
            if valor > self.sl_mascara.maximum():
                valor -= 2
            self.sl_mascara.blockSignals(True)
            self.sl_mascara.setValue(valor)
            self.sl_mascara.blockSignals(False)
        return valor

    def _on_mascara(self, valor):
        if valor % 2 == 0:
            valor += 1
            if valor > self.sl_mascara.maximum():
                valor -= 2
            self.sl_mascara.blockSignals(True)
            self.sl_mascara.setValue(valor)
            self.sl_mascara.blockSignals(False)
        kernel = self._kernel_actual()
        self.lbl_kernel.setText(f"Mascara {kernel}x{kernel}")
        self._actualizar_resumen_flujo()

    def _on_dominio_suavizado(self, dominio):
        es_espacial = dominio == "Espacial"
        self.lbl_suavizado.setVisible(es_espacial)
        self.combo_suavizado.setVisible(es_espacial)
        self.lbl_kernel.setVisible(es_espacial)
        self.sl_mascara.setVisible(es_espacial)
        self.lbl_frecuencia_suavizado.setVisible(not es_espacial)
        self.combo_frecuencia_suavizado.setVisible(not es_espacial)
        self.lbl_d0_suavizado.setVisible(not es_espacial)
        self.lbl_d0_suavizado_valor.setVisible(not es_espacial)
        self.sl_d0_suavizado.setVisible(not es_espacial)

    def _on_dominio_acentuado(self, dominio):
        es_espacial = dominio == "Espacial"
        self.lbl_acentuado.setVisible(es_espacial)
        self.combo_acentuado.setVisible(es_espacial)
        self.lbl_frecuencia_acentuado.setVisible(not es_espacial)
        self.combo_frecuencia_acentuado.setVisible(not es_espacial)
        self.lbl_d0_acentuado.setVisible(not es_espacial)
        self.lbl_d0_acentuado_valor.setVisible(not es_espacial)
        self.sl_d0_acentuado.setVisible(not es_espacial)
        self._actualizar_estado_factor_acentuado()

    def _actualizar_estado_factor_acentuado(self):
        if not hasattr(self, "combo_dominio_acentuado"):
            return
        es_frecuencia = self.combo_dominio_acentuado.currentText() == "Frecuencial"
        usa_high_boost = (
            self.combo_frecuencia_acentuado.currentText() == "High-Boost"
            if es_frecuencia
            else self.combo_acentuado.currentText() == "High-Boost"
        )
        self.lbl_factor_acentuado.setVisible(usa_high_boost)
        self.lbl_factor_acentuado_valor.setVisible(usa_high_boost)
        self.sl_factor_acentuado.setVisible(usa_high_boost)

    def _actualizar_resumen_flujo(self):
        if not hasattr(self, "lbl_flujo"):
            return

        suavizado = getattr(self, "combo_suavizado", None)
        dominio_suavizado = getattr(self, "combo_dominio_suavizado", None)
        acentuado = getattr(self, "combo_acentuado", None)
        dominio_acentuado = getattr(self, "combo_dominio_acentuado", None)
        gradiente = getattr(self, "combo_gradiente", None)
        min_area = getattr(self, "spin_min_area", None)
        max_area = getattr(self, "spin_max_area", None)

        if not all([suavizado, dominio_suavizado, acentuado, dominio_acentuado, gradiente, min_area, max_area]):
            return

        ruido = getattr(self, "sl_ruido", None)
        umbral = getattr(self, "sl_umbral", None)
        d0_suavizado = getattr(self, "sl_d0_suavizado", None)
        d0_acentuado = getattr(self, "sl_d0_acentuado", None)
        freq_suavizado = getattr(self, "combo_frecuencia_suavizado", None)
        freq_acentuado = getattr(self, "combo_frecuencia_acentuado", None)
        factor = getattr(self, "sl_factor_acentuado", None)
        ruido_txt = f"{ruido.value()}%" if ruido else "0%"
        umbral_txt = str(umbral.value()) if umbral else "128"
        d0_suavizado_txt = str(d0_suavizado.value()) if d0_suavizado else "45"
        d0_acentuado_txt = str(d0_acentuado.value()) if d0_acentuado else "45"
        max_area_txt = str(max_area.value()) if max_area.value() > 0 else "sin límite"
        suavizado_txt = suavizado.currentText()
        if dominio_suavizado.currentText() == "Frecuencial":
            suavizado_txt = f"{freq_suavizado.currentText()} D0 {d0_suavizado_txt}"
        acentuado_txt = acentuado.currentText()
        if dominio_acentuado.currentText() == "Frecuencial":
            acentuado_txt = f"{freq_acentuado.currentText()} D0 {d0_acentuado_txt}"
        if (
            (dominio_acentuado.currentText() == "Espacial" and acentuado.currentText() == "High-Boost")
            or (dominio_acentuado.currentText() == "Frecuencial" and freq_acentuado.currentText() == "High-Boost")
        ):
            acentuado_txt = f"{acentuado_txt} A {factor.value() / 10:.1f}"

        self.lbl_flujo.setText(
            "Activos: "
            f"R {ruido_txt} | "
            f"S {dominio_suavizado.currentText()}/{suavizado_txt} | "
            f"A {dominio_acentuado.currentText()}/{acentuado_txt} | "
            f"G {gradiente.currentText()} | "
            f"U {umbral_txt} | "
            f"Á {min_area.value()}-{max_area_txt}"
        )

    def _botones_procesamiento(self):
        botones = [self.btn_procesar]
        for nombre in (
            "btn_aplicar_pre",
            "btn_aplicar_suavizado",
            "btn_aplicar_acentuado",
            "btn_aplicar_regiones",
        ):
            if hasattr(self, nombre):
                botones.append(getattr(self, nombre))
        return botones

    def _set_procesando(self, procesando):
        hay_imagen = self.img_rgb is not None
        self.btn_cargar.setEnabled(not procesando)
        self.btn_limpiar.setEnabled(not procesando)
        for boton in self._botones_procesamiento():
            boton.setEnabled(hay_imagen and not procesando)
        if hasattr(self, "_actualizar_estado_guardado"):
            self._actualizar_estado_guardado(procesando)

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import modelo


def to_rgb_format(img_np):
    """Asegura que la imagen se muestre como RGB en matplotlib."""
    if img_np.ndim == 2:
        return np.stack((img_np, img_np, img_np), axis=2)
    return img_np


def estilo_slider(color):
    return f"""
    QSlider::groove:horizontal {{
        border: none;
        height: 6px;
        background: #A0A7B4;
        border-radius: 4px;
    }}
    QSlider::sub-page:horizontal {{
        background: {color};
        border-radius: 4px;
    }}
    QSlider::add-page:horizontal {{
        background: #A0A7B4;
        border-radius: 4px;
    }}
    QSlider::handle:horizontal {{
        background: white;
        border: 2px solid {color};
        width: 14px;
        margin: -5px 0;
        border-radius: 9px;
    }}
    """


APP_STYLE = """
QMainWindow { background: #EEF2F7; }
QWidget { color: #253040; font-size: 13px; }
QFrame#Sidebar {
    background: #15202B;
    border: none;
}
QFrame#Panel {
    background: white;
    border: 1px solid #D8DCE4;
    border-radius: 16px;
}
QFrame#ControlCard {
    background: #1D2B38;
    border: 1px solid #314353;
    border-radius: 18px;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: #15202B;
    width: 10px;
    margin: 6px 0;
}
QScrollBar::handle:vertical {
    background: #304252;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QTabWidget::pane {
    border: 1px solid #D8DCE4;
    background: white;
    border-radius: 14px;
}
QTabBar::tab {
    background: #DDE7F1;
    color: #304055;
    border: 1px solid #C7D4E1;
    padding: 10px 16px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 700;
}
QTabBar::tab:selected {
    background: white;
    border-bottom-color: white;
}
QLabel#MainTitle {
    font-size: 18px;
    font-weight: 800;
    color: #222B38;
}
QLabel#SidebarTitle {
    font-size: 18px;
    font-weight: 800;
    color: #F5F7FA;
}
QLabel#SidebarText {
    color: #AFC1D1;
    line-height: 1.3;
}
QLabel#SectionTitle {
    font-size: 13px;
    font-weight: 700;
    color: #2F3B4C;
}
QLabel#Muted {
    color: #5F6C7E;
}
QLabel#SidebarMuted {
    color: #9EB1C4;
}
QLabel#CardTitle {
    font-size: 14px;
    font-weight: 800;
    color: #F4F7FB;
}
QLabel#CardHint {
    color: #95A8BA;
    line-height: 1.25;
}
QLabel#ValueBadge {
    background: #2B4053;
    color: #F7FBFF;
    border-radius: 10px;
    padding: 5px 10px;
    font-weight: 700;
}
QPushButton {
    background: #E56B4A;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 24px;
    padding: 10px 14px;
    font-weight: 700;
}
QPushButton:hover { background: #D55D3C; }
QPushButton:disabled { background: #9EA8B4; }
QPushButton#ActionButton {
    background: #24A187;
}
QPushButton#ActionButton:hover {
    background: #1D8A73;
}
QPushButton#SecondaryButton {
    background: #357ABD;
}
QPushButton#SecondaryButton:hover {
    background: #2D68A3;
}
QPushButton#DangerButton {
    background: #C24E4E;
}
QPushButton#DangerButton:hover {
    background: #AA4242;
}
QComboBox {
    background: #F1F5F9;
    color: #223142;
    border: 1px solid #D2DCE6;
    border-radius: 10px;
    min-height: 20px;
    padding: 8px 10px;
}
QComboBox::drop-down {
    color: white;
    border: none;
}
"""


class WorkerProcesoCompleto(QThread):
    terminado = Signal(dict)
    error = Signal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        try:
            print("[pipeline] Iniciando procesamiento completo...")
            img_rgb = self.args["img_rgb"]
            ruido = self.args["ruido"]
            tam_mascara = self.args["mascara"]
            d0 = self.args["d0"]
            tipo_filtro = self.args["tipo_filtro"]
            umbral = self.args["umbral"]

            img_gris = modelo.convertir_a_grises(img_rgb)
            hist_gris = modelo.calcular_histograma_grises(img_gris)
            print("[pipeline] Grises listos.")

            img_normalizada = modelo.normalizar_histograma_grises(img_gris)
            hist_normalizada = modelo.calcular_histograma_grises(img_normalizada)
            print("[pipeline] Normalizacion lista.")

            img_binaria = modelo.binarizar_imagen(img_normalizada, umbral)
            img_ruido = modelo.agregar_ruido_sal_pimienta(img_binaria, ruido)
            print("[pipeline] Binarizado y ruido listos.")

            if tipo_filtro == "Media":
                resultado_espacial = modelo.filtro_media(img_ruido, tam_mascara)
            elif tipo_filtro == "Mediana":
                resultado_espacial = modelo.filtro_mediana(img_ruido, tam_mascara)
            else:
                resultado_espacial = modelo.filtro_moda(img_ruido, tam_mascara)
            print(f"[pipeline] Filtro espacial listo: {tipo_filtro}.")

            resultado_frecuencia = modelo.filtro_frecuencia_gaussiano(img_ruido, d0)
            diagnostico_frecuencia = modelo.diagnostico_frecuencia(img_ruido, d0)
            mapa_cambio = modelo.diferencia_absoluta_manual(img_ruido, resultado_espacial)
            print("[pipeline] Frecuencia y mapa de cambio listos.")

            self.terminado.emit(
                {
                    "original": img_rgb,
                    "gris": img_gris,
                    "hist_gris": hist_gris,
                    "normalizada": img_normalizada,
                    "hist_normalizada": hist_normalizada,
                    "binaria": img_binaria,
                    "ruido": img_ruido,
                    "espacial": resultado_espacial,
                    "frecuencia": resultado_frecuencia,
                    "mapa_cambio": mapa_cambio,
                    "espectro_original": diagnostico_frecuencia["espectro_original"],
                    "mascara_frecuencia": diagnostico_frecuencia["mascara"],
                    "espectro_filtrado": diagnostico_frecuencia["espectro_filtrado"],
                    "tipo_filtro": tipo_filtro,
                    "mascara": tam_mascara,
                    "d0": d0,
                    "umbral": umbral,
                    "ruido_porcentaje": int(round(ruido * 100)),
                }
            )
            print("[pipeline] Todo el proceso termino bien.")
        except Exception as exc:
            print(f"[pipeline] Error: {exc}")
            self.error.emit(str(exc))


class ComboSoloDropdown(QComboBox):
    """Evita cambios accidentales con la rueda del mouse."""

    def wheelEvent(self, event):
        event.ignore()


class CanvasResultados(FigureCanvas):
    def __init__(self, items, filas, columnas, parent=None):
        self.fig = Figure(figsize=(12, 7), facecolor="white")
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.08, wspace=0.22, hspace=0.30)
        super().__init__(self.fig)
        self.setParent(parent)

        self.items = items
        self.axes = [
            self.fig.add_subplot(filas, columnas, indice + 1)
            for indice in range(len(items))
        ]

    def actualizar_titulos(self, titulos):
        for item, titulo in zip(self.items, titulos):
            item["titulo"] = titulo

    def actualizar(self, datos=None):
        datos = datos or {}

        for ax, item in zip(self.axes, self.items):
            ax.clear()
            ax.set_facecolor("#F8FAFD")
            ax.set_title(item["titulo"], fontsize=11, fontweight="bold", color="#243244")

            if item["tipo"] == "imagen":
                self._dibujar_imagen(ax, datos.get(item["clave"]))
            else:
                self._dibujar_histograma(ax, item, datos)

        self.draw_idle()

    def _dibujar_imagen(self, ax, imagen):
        ax.set_xticks([])
        ax.set_yticks([])

        if imagen is None:
            ax.text(
                0.5,
                0.5,
                "Sin datos",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="#8090A0",
                fontweight="bold",
            )
            return

        ax.imshow(to_rgb_format(imagen), interpolation="nearest")

    def _dibujar_histograma(self, ax, item, datos):
        histograma = datos.get(item["clave"])
        ax.set_facecolor("white")

        if histograma is None:
            ax.text(
                0.5,
                0.5,
                "Sin histograma",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="#8090A0",
                fontweight="bold",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return

        x = np.arange(256)
        ax.bar(x, histograma, width=1.0, color=item.get("color", "#4B80D1"), edgecolor=item.get("color", "#4B80D1"))
        ax.set_xlim(0, 255)
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.tick_params(axis="x", labelsize=8, colors="#5A6678")
        ax.tick_params(axis="y", labelsize=8, colors="#5A6678")
        ax.grid(axis="y", color="#DCE3EB", alpha=0.9, linewidth=0.8)

        linea_clave = item.get("linea_clave")
        if linea_clave and linea_clave in datos:
            ax.axvline(datos[linea_clave], color="#D54F45", linewidth=1.8, linestyle="--")


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesamiento de Imagenes: Dominio Espacial y Frecuencia")
        self.resize(1520, 920)
        self.setStyleSheet(APP_STYLE)

        self.img_rgb = None
        self.max_mascara = 3
        self.worker = None

        self._build_ui()
        self._mostrar_placeholders()

    def _build_ui(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(360)

        sidebar_host_layout = QVBoxLayout(sidebar)
        sidebar_host_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_host_layout.setSpacing(0)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(24, 24, 24, 24)
        sidebar_layout.setSpacing(14)

        sidebar_titulo = QLabel("Laboratorio de filtros")
        sidebar_titulo.setObjectName("SidebarTitle")
        sidebar_layout.addWidget(sidebar_titulo)
        
        botones_layout = QHBoxLayout()
        botones_layout.setSpacing(10)

        self.btn_cargar = QPushButton("Cargar imagen")
        self.btn_cargar.setObjectName("SecondaryButton")
        self.btn_cargar.clicked.connect(self._cargar_imagen)
        botones_layout.addWidget(self.btn_cargar)

        self.btn_limpiar = QPushButton("Limpiar")
        self.btn_limpiar.setObjectName("DangerButton")
        self.btn_limpiar.clicked.connect(self._limpiar_todo)
        botones_layout.addWidget(self.btn_limpiar)

        sidebar_layout.addLayout(botones_layout)

        self.lbl_path = QLabel("Ninguna imagen cargada")
        self.lbl_path.setObjectName("SidebarMuted")
        self.lbl_path.setWordWrap(True)
        sidebar_layout.addWidget(self.lbl_path)

        self.lbl_size = QLabel("")
        self.lbl_size.setObjectName("SidebarMuted")
        sidebar_layout.addWidget(self.lbl_size)

        self._agregar_bloque_preprocesamiento(sidebar_layout)
        self._agregar_bloque_ruido(sidebar_layout)
        self._agregar_bloque_filtro_espacial(sidebar_layout)
        self._agregar_bloque_filtro_frecuencia(sidebar_layout)

        self.btn_procesar = QPushButton("Procesar todo")
        self.btn_procesar.setObjectName("ActionButton")
        self.btn_procesar.setEnabled(False)
        self.btn_procesar.clicked.connect(self._procesar_todo)
        sidebar_layout.addWidget(self.btn_procesar)

        self.lbl_estado = QLabel("Carga una imagen para empezar.")
        self.lbl_estado.setObjectName("SidebarMuted")
        self.lbl_estado.setWordWrap(True)
        sidebar_layout.addWidget(self.lbl_estado)

        sidebar_layout.addStretch(1)
        sidebar_scroll.setWidget(sidebar_content)
        sidebar_host_layout.addWidget(sidebar_scroll)

        contenido = QWidget()
        contenido_layout = QVBoxLayout(contenido)
        contenido_layout.setContentsMargins(18, 18, 18, 18)
        contenido_layout.setSpacing(12)

        header_panel = QFrame()
        header_panel.setObjectName("Panel")
        header_layout = QVBoxLayout(header_panel)
        header_layout.setContentsMargins(18, 14, 18, 14)
        header_layout.setSpacing(8)

        titulo = QLabel("Procesamiento digital de imagenes")
        titulo.setObjectName("MainTitle")
        header_layout.addWidget(titulo)

        subtitulo = QLabel(
            "La interfaz separa el trabajo en tres etapas: preprocesamiento, filtrado espacial por convolucion/manual y filtrado gaussiano en frecuencia."
        )
        subtitulo.setObjectName("Muted")
        subtitulo.setWordWrap(True)
        header_layout.addWidget(subtitulo)

        contenido_layout.addWidget(header_panel)

        canvas_panel = QFrame()
        canvas_panel.setObjectName("Panel")
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(14, 14, 14, 14)
        canvas_layout.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._crear_tab_preprocesamiento(), "Preprocesamiento")
        self.tabs.addTab(self._crear_tab_espacial(), "Dominio espacial")
        self.tabs.addTab(self._crear_tab_frecuencia(), "Dominio de frecuencia")

        canvas_layout.addWidget(self.tabs)
        contenido_layout.addWidget(canvas_panel, 1)

        root_layout.addWidget(sidebar)
        root_layout.addWidget(contenido, 1)
        self.setCentralWidget(root)

    def _crear_tab_preprocesamiento(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        descripcion = QLabel(
            "Secuencia base: imagen RGB, escala de grises, ecualizacion y binarizacion final."
        )
        descripcion.setObjectName("Muted")
        descripcion.setWordWrap(True)
        layout.addWidget(descripcion)

        self.canvas_preprocesamiento = CanvasResultados(
            [
                {"tipo": "imagen", "clave": "original", "titulo": "Imagen original RGB"},
                {"tipo": "imagen", "clave": "gris", "titulo": "Escala de grises"},
                {"tipo": "imagen", "clave": "normalizada", "titulo": "Imagen ecualizada"},
                {"tipo": "imagen", "clave": "binaria", "titulo": "Imagen binarizada"},
            ],
            2,
            2,
            self,
        )
        layout.addWidget(self.canvas_preprocesamiento)
        return tab

    def _crear_tab_espacial(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        descripcion = QLabel(
            "Se aplica ruido sal y pimienta sobre la binaria y luego un filtro espacial seleccionado entre media, mediana o moda usando una mascara impar."
        )
        descripcion.setObjectName("Muted")
        descripcion.setWordWrap(True)
        layout.addWidget(descripcion)

        self.canvas_espacial = CanvasResultados(
            [
                {"tipo": "imagen", "clave": "binaria", "titulo": "Base binarizada"},
                {"tipo": "imagen", "clave": "ruido", "titulo": "Ruido sal y pimienta"},
                {"tipo": "imagen", "clave": "espacial", "titulo": "Resultado espacial"},
                {"tipo": "imagen", "clave": "mapa_cambio", "titulo": "Mapa de cambio"},
            ],
            2,
            2,
            self,
        )
        layout.addWidget(self.canvas_espacial)
        return tab

    def _crear_tab_frecuencia(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        descripcion = QLabel(
            "Dominio de frecuencia con FFT y filtro gaussiano pasa bajas. Se muestran el espectro, la mascara y la reconstruccion final."
        )
        descripcion.setObjectName("Muted")
        descripcion.setWordWrap(True)
        layout.addWidget(descripcion)

        self.canvas_frecuencia = CanvasResultados(
            [
                {"tipo": "imagen", "clave": "binaria", "titulo": "Base binarizada"},
                {"tipo": "imagen", "clave": "ruido", "titulo": "Entrada con ruido"},
                {"tipo": "imagen", "clave": "espectro_original", "titulo": "Espectro FFT"},
                {"tipo": "imagen", "clave": "mascara_frecuencia", "titulo": "Mascara gaussiana"},
                {"tipo": "imagen", "clave": "espectro_filtrado", "titulo": "Espectro filtrado"},
                {"tipo": "imagen", "clave": "frecuencia", "titulo": "Imagen reconstruida"},
            ],
            2,
            3,
            self,
        )
        layout.addWidget(self.canvas_frecuencia)
        return tab

    def _crear_tarjeta_control(self, titulo, ayuda=None):
        tarjeta = QFrame()
        tarjeta.setObjectName("ControlCard")
        layout = QVBoxLayout(tarjeta)
        layout.setContentsMargins(18, 16, 18, 16)
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

    def _agregar_bloque_preprocesamiento(self, layout_principal):
        tarjeta, bloque = self._crear_tarjeta_control(
            "Preprocesamiento"
        )

        self.lbl_umbral = QLabel("Umbral 127")
        self.lbl_umbral.setObjectName("ValueBadge")
        bloque.addWidget(self.lbl_umbral, alignment=Qt.AlignLeft)

        self.sl_umbral = QSlider(Qt.Horizontal)
        self.sl_umbral.setRange(0, 255)
        self.sl_umbral.setValue(127)
        self.sl_umbral.setStyleSheet(estilo_slider("#E56B4A"))
        self.sl_umbral.valueChanged.connect(self._actualizar_lbl_umbral)
        bloque.addWidget(self.sl_umbral)

        layout_principal.addWidget(tarjeta)

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
        bloque.addWidget(self.sl_ruido)

        layout_principal.addWidget(tarjeta)

    def _agregar_bloque_filtro_espacial(self, layout_principal):
        tarjeta, bloque = self._crear_tarjeta_control("Dominio espacial")

        self.combo_filtro = ComboSoloDropdown()
        self.combo_filtro.addItems(["Media", "Mediana", "Moda"])
        bloque.addWidget(self.combo_filtro)

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

        layout_principal.addWidget(tarjeta)

    def _agregar_bloque_filtro_frecuencia(self, layout_principal):
        tarjeta, bloque = self._crear_tarjeta_control("Dominio de frecuencia")

        self.lbl_d0 = QLabel("D0 45 px")
        self.lbl_d0.setObjectName("ValueBadge")
        bloque.addWidget(self.lbl_d0, alignment=Qt.AlignLeft)

        self.sl_d0 = QSlider(Qt.Horizontal)
        self.sl_d0.setRange(1, 180)
        self.sl_d0.setValue(45)
        self.sl_d0.setStyleSheet(estilo_slider("#F2B15E"))
        self.sl_d0.valueChanged.connect(self._actualizar_lbl_d0)
        bloque.addWidget(self.sl_d0)

        layout_principal.addWidget(tarjeta)

    def _actualizar_lbl_umbral(self, valor):
        self.lbl_umbral.setText(f"Umbral {valor}")

    def _actualizar_lbl_ruido(self, valor):
        self.lbl_ruido.setText(f"{valor} %")

    def _actualizar_lbl_d0(self, valor):
        self.lbl_d0.setText(f"D0 {valor} px")

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

        self.sl_d0.blockSignals(True)
        self.sl_d0.setRange(1, maximo)
        self.sl_d0.setValue(sugerido)
        self.sl_d0.blockSignals(False)
        self._actualizar_lbl_d0(sugerido)

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

    def _mostrar_placeholders(self):
        self.canvas_preprocesamiento.actualizar()
        self.canvas_espacial.actualizar()
        self.canvas_frecuencia.actualizar()

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
            # Cargar esto limpio ayuda bastante para revisar las demas etapas.
            self.img_rgb = modelo.cargar_imagen(ruta)
            alto, ancho = self.img_rgb.shape[:2]
            print(f"[ui] Imagen cargada en la interfaz: {Path(ruta).name}")

            self.lbl_path.setText(Path(ruta).name)
            self.lbl_size.setText(f"{ancho} x {alto} px")

            self.max_mascara = modelo.calcular_tamano_mascara_maximo(self.img_rgb)
            self._actualizar_opciones_kernel(self.max_mascara)
            self._actualizar_opciones_d0(alto, ancho)

            self.btn_procesar.setEnabled(True)
            self.lbl_estado.setText("Imagen lista. Ajusta parametros y pulsa Procesar todo.")

            self.canvas_preprocesamiento.actualizar({"original": self.img_rgb})
            self.canvas_espacial.actualizar()
            self.canvas_frecuencia.actualizar()
        except Exception as exc:
            self.lbl_estado.setText(f"Error al cargar la imagen: {exc}")

    def _limpiar_todo(self):
        if self.worker and self.worker.isRunning():
            return

        print("[ui] Limpiando resultados y reiniciando controles...")
        self.img_rgb = None
        self.lbl_path.setText("Ninguna imagen cargada")
        self.lbl_size.setText("")
        self.lbl_estado.setText("Carga una imagen para empezar.")

        self.sl_umbral.setValue(127)
        self.sl_ruido.setValue(5)
        self.sl_mascara.setValue(3)
        self.sl_d0.setValue(45)
        self.combo_filtro.setCurrentText("Media")

        self.btn_procesar.setEnabled(False)
        self._mostrar_placeholders()

    def _procesar_todo(self):
        if self.img_rgb is None:
            return

        print("[ui] Lanzando procesamiento desde la interfaz...")
        self.btn_procesar.setEnabled(False)
        self.btn_cargar.setEnabled(False)
        self.btn_limpiar.setEnabled(False)
        self.lbl_estado.setText("Procesando preprocesamiento, dominio espacial y frecuencia...")

        args = {
            "img_rgb": self.img_rgb.copy(),
            "ruido": self.sl_ruido.value() / 100.0,
            "mascara": self._kernel_actual(),
            "d0": self.sl_d0.value(),
            "tipo_filtro": self.combo_filtro.currentText(),
            "umbral": self.sl_umbral.value(),
        }

        self.worker = WorkerProcesoCompleto(args)
        self.worker.terminado.connect(self._on_proceso_listo)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_proceso_listo(self, resultado):
        # Esta es la parte bonita: ya se ve todo el recorrido completo.
        print("[ui] Actualizando paneles con los resultados...")
        self.canvas_preprocesamiento.actualizar(resultado)

        filtro = resultado["tipo_filtro"]
        mascara = resultado["mascara"]
        self.canvas_espacial.actualizar_titulos(
            [
                "Base binarizada",
                f"Ruido sal y pimienta ({resultado['ruido_porcentaje']} %)",
                f"Resultado {filtro} ({mascara}x{mascara})",
                "Mapa de cambio",
            ]
        )
        self.canvas_espacial.actualizar(resultado)

        self.canvas_frecuencia.actualizar_titulos(
            [
                "Base binarizada",
                f"Entrada con ruido ({resultado['ruido_porcentaje']} %)",
                "Espectro FFT",
                f"Mascara gaussiana (D0={resultado['d0']})",
                "Espectro filtrado",
                "Imagen reconstruida",
            ]
        )
        self.canvas_frecuencia.actualizar(resultado)

        self.lbl_estado.setText(
            f"Proceso Completado. Espacial: {filtro} con mascara {mascara}x{mascara}. Frecuencia: gaussiano pasa bajas con D0={resultado['d0']} px."
        )

        self.btn_procesar.setEnabled(True)
        self.btn_cargar.setEnabled(True)
        self.btn_limpiar.setEnabled(True)
        self.worker = None
        print("[ui] Interfaz lista para otra prueba.")

    def _on_error(self, mensaje):
        print(f"[ui] Se mostro un error en pantalla: {mensaje}")
        self.lbl_estado.setText(f"Error durante el proceso: {mensaje}")
        self.btn_procesar.setEnabled(self.img_rgb is not None)
        self.btn_cargar.setEnabled(True)
        self.btn_limpiar.setEnabled(True)
        self.worker = None

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
    QSpinBox,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from aplicacion import ParametrosProcesamiento, ProcesadorImagen
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
QScrollArea#TabScrollArea {
    background: white;
}
QWidget#TabScrollContent {
    background: white;
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
    background: #243648;
    color: #F0F5FA;
    border: 2px solid #4A7FA5;
    border-radius: 10px;
    min-height: 20px;
    padding: 8px 10px;
    font-weight: 700;
}
QComboBox::drop-down {
    color: white;
    border: none;
}
QComboBox QAbstractItemView {
    background: #1D2B38;
    color: #F0F5FA;
    selection-background-color: #357ABD;
    selection-color: white;
    border: 1px solid #4A7FA5;
    outline: none;
}
"""


class WorkerProcesoCompleto(QThread):
    terminado = Signal(object)
    error = Signal(str)

    def __init__(self, parametros, procesador=None):
        super().__init__()
        self.parametros = parametros
        self.procesador = procesador or ProcesadorImagen()

    def run(self):
        try:
            resultado = self.procesador.ejecutar(self.parametros)
            self.terminado.emit(resultado)
        except Exception as exc:
            print(f"[pipeline] Error: {exc}")
            self.error.emit(str(exc))


class ComboSoloDropdown(QComboBox):
    """Evita cambios accidentales con la rueda del mouse."""

    def wheelEvent(self, event):
        event.ignore()


class CanvasResultados(FigureCanvas):
    def __init__(self, items, filas, columnas, parent=None):
        altura = self._calcular_altura_canvas(len(items), filas, columnas)
        self.fig = Figure(figsize=(12, altura / 100.0), facecolor="white")
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.10, wspace=0.18, hspace=0.24)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background: white;")
        self.setMinimumHeight(altura)
        self.setMaximumHeight(altura)

        self.items = items
        self.axes = [
            self.fig.add_subplot(filas, columnas, indice + 1)
            for indice in range(len(items))
        ]

    def _calcular_altura_canvas(self, total_items, filas, columnas):
        if total_items == 1:
            return 170
        if filas == 1 and columnas == 2:
            return 215
        if filas == 1 and columnas == 3:
            return 205
        if filas == 2 and columnas == 2:
            return 390
        return 340

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
        self.ultimo_resultado = None

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
        self._agregar_bloque_ruido(sidebar_layout)
        self._agregar_bloque_suavizado(sidebar_layout)
        self._agregar_bloque_acentuado(sidebar_layout)
        self._agregar_bloque_binarizacion(sidebar_layout)

        self.btn_procesar = QPushButton("Aplicar")
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
            "Pipeline por etapas."
        )
        subtitulo.setObjectName("Muted")
        subtitulo.setWordWrap(True)
        header_layout.addWidget(subtitulo)

        self.lbl_flujo = QLabel("")
        self.lbl_flujo.setObjectName("Muted")
        self.lbl_flujo.setWordWrap(True)
        header_layout.addWidget(self.lbl_flujo)

        contenido_layout.addWidget(header_panel)

        canvas_panel = QFrame()
        canvas_panel.setObjectName("Panel")
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(14, 14, 14, 14)
        canvas_layout.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._crear_tab_preprocesamiento(), "Escala de grises")
        self.tabs.addTab(self._crear_tab_suavizado(), "Suavizado")
        self.tabs.addTab(self._crear_tab_acentuado(), "Acentuado")
        self.tabs.addTab(self._crear_tab_binarizacion(), "Binarización")
        self.tabs.addTab(self._crear_tab_gradiente(), "Gradiente")
        self.tabs.addTab(self._crear_tab_regiones(), "Regiones")

        canvas_layout.addWidget(self.tabs)
        contenido_layout.addWidget(canvas_panel, 1)

        root_layout.addWidget(sidebar)
        root_layout.addWidget(contenido, 1)
        self.setCentralWidget(root)
        self._actualizar_resumen_flujo()

    def _crear_tab_preprocesamiento(self):
        tab, layout = self._crear_tab_scrollable()

        self.canvas_preprocesamiento = CanvasResultados(
            [
                {"tipo": "imagen", "clave": "original",    "titulo": "Imagen original RGB"},
                {"tipo": "imagen", "clave": "gris",        "titulo": "Escala de grises"},
                {"tipo": "imagen", "clave": "normalizada", "titulo": "Histograma normalizado"},
                {"tipo": "imagen", "clave": "ruido",       "titulo": "Con ruido sal y pimienta"},
            ],
            2,
            2,
            self,
        )
        layout.addWidget(self.canvas_preprocesamiento)
        return tab

    def _crear_tab_suavizado(self):
        tab, layout = self._crear_tab_scrollable()

        self.canvas_suavizado = CanvasResultados(
            [
                {"tipo": "imagen", "clave": "ruido",     "titulo": "Entrada"},
                {"tipo": "imagen", "clave": "suavizada", "titulo": "Salida"},
            ],
            1,
            2,
            self,
        )
        layout.addWidget(self.canvas_suavizado)
        return tab

    def _crear_tab_acentuado(self):
        tab, layout = self._crear_tab_scrollable()

        self.canvas_acentuado = CanvasResultados(
            [
                {"tipo": "imagen", "clave": "suavizada", "titulo": "Entrada"},
                {"tipo": "imagen", "clave": "acentuada", "titulo": "Salida"},
            ],
            1,
            2,
            self,
        )
        layout.addWidget(self.canvas_acentuado)
        return tab

    def _crear_tab_gradiente(self):
        tab, contenido_layout = self._crear_tab_scrollable()

        self.canvas_gradiente = CanvasResultados(
            [
                {"tipo": "imagen", "clave": "binaria_acentuada", "titulo": "Binarizada post-acentuado"},
                {"tipo": "imagen", "clave": "gradiente_final",   "titulo": "Gradiente seleccionado"},
                {"tipo": "imagen", "clave": "binaria",           "titulo": "Bordes binarios finales"},
            ],
            1,
            3,
            self,
        )
        contenido_layout.addWidget(self.canvas_gradiente)

        subtitulo_componentes = QLabel("Componentes del gradiente.")
        subtitulo_componentes.setObjectName("Muted")
        subtitulo_componentes.setWordWrap(True)
        contenido_layout.addWidget(subtitulo_componentes)

        self.canvas_componentes_gradiente = CanvasResultados(
            [
                {"tipo": "imagen", "clave": "comp_0", "titulo": "Componente 1"},
                {"tipo": "imagen", "clave": "comp_1", "titulo": "Componente 2"},
                {"tipo": "imagen", "clave": "comp_2", "titulo": "Componente 3"},
            ],
            1,
            3,
            self,
        )
        contenido_layout.addWidget(self.canvas_componentes_gradiente)
        contenido_layout.addStretch(1)

        return tab

    def _crear_tab_binarizacion(self):
        tab, layout = self._crear_tab_scrollable()

        self.canvas_binarizacion = CanvasResultados(
            [
                {"tipo": "imagen", "clave": "acentuada",         "titulo": "Acentuado (entrada)"},
                {"tipo": "imagen", "clave": "binaria_acentuada", "titulo": "Imagen binaria (salida)"},
            ],
            1, 2, self,
        )
        layout.addWidget(self.canvas_binarizacion)

        return tab

    def _crear_tab_regiones(self):
        tab, layout = self._crear_tab_scrollable()

        self.lbl_regiones_info = QLabel("Sin datos.")
        self.lbl_regiones_info.setObjectName("Muted")
        self.lbl_regiones_info.setWordWrap(True)
        layout.addWidget(self.lbl_regiones_info)

        self.canvas_regiones = CanvasResultados(
            [
                {"tipo": "imagen", "clave": "binaria", "titulo": "Gradiente binario (entrada detección)"},
                {"tipo": "imagen", "clave": "bboxes",  "titulo": "Regiones detectadas"},
            ],
            1,
            2,
            self,
        )
        layout.addWidget(self.canvas_regiones)

        self.canvas_recortes = CanvasResultados(
            [
                {"tipo": "imagen", "clave": "recortes_tira", "titulo": "Submatrices normalizadas — entrada al clasificador"},
            ],
            1,
            1,
            self,
        )
        layout.addWidget(self.canvas_recortes)
        return tab

    def _crear_tab_scrollable(self):
        tab = QWidget()
        tab.setStyleSheet("background: white;")
        root = QVBoxLayout(tab)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("TabScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: white; border: none;")
        scroll.viewport().setStyleSheet("background: white;")

        contenido = QWidget()
        contenido.setObjectName("TabScrollContent")
        contenido.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QVBoxLayout(contenido)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        scroll.setWidget(contenido)
        root.addWidget(scroll)
        return tab, layout

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

        lbl_suavizado = QLabel("Suavizado")
        lbl_suavizado.setObjectName("CardHint")
        bloque.addWidget(lbl_suavizado)

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

        lbl_d0_suavizado = QLabel("D0")
        lbl_d0_suavizado.setObjectName("CardHint")
        bloque.addWidget(lbl_d0_suavizado)

        self.lbl_d0 = QLabel("D0 45 px")
        self.lbl_d0.setObjectName("ValueBadge")
        bloque.addWidget(self.lbl_d0, alignment=Qt.AlignLeft)

        self.sl_d0 = QSlider(Qt.Horizontal)
        self.sl_d0.setRange(1, 180)
        self.sl_d0.setValue(45)
        self.sl_d0.setStyleSheet(estilo_slider("#F2B15E"))
        self.sl_d0.valueChanged.connect(self._actualizar_lbl_d0)
        self.sl_d0.valueChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.sl_d0)

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

        lbl_acentuado = QLabel("Operador")
        lbl_acentuado.setObjectName("CardHint")
        bloque.addWidget(lbl_acentuado)

        self.combo_acentuado = ComboSoloDropdown()
        self.combo_acentuado.addItems(["Roberts", "Prewitt", "Sobel", "Laplaciano"])
        self.combo_acentuado.currentTextChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.combo_acentuado)

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
        self.combo_gradiente.addItems(["Roberts", "Prewitt", "Sobel", "Laplaciano"])
        self.combo_gradiente.currentTextChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.combo_gradiente)

        lbl_min_area = QLabel("Área mínima")
        lbl_min_area.setObjectName("CardHint")
        bloque.addWidget(lbl_min_area)

        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(1, 999999)
        self.spin_min_area.setValue(50)
        self.spin_min_area.valueChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.spin_min_area)

        lbl_max_area = QLabel("Área máxima (0 = sin límite)")
        lbl_max_area.setObjectName("CardHint")
        bloque.addWidget(lbl_max_area)

        self.spin_max_area = QSpinBox()
        self.spin_max_area.setRange(0, 999999)
        self.spin_max_area.setValue(0)
        self.spin_max_area.valueChanged.connect(lambda _: self._actualizar_resumen_flujo())
        bloque.addWidget(self.spin_max_area)

        layout_principal.addWidget(tarjeta)

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
        self.combo_suavizado.setEnabled(es_espacial)
        self.sl_mascara.setEnabled(es_espacial)
        self.lbl_kernel.setEnabled(es_espacial)
        self._actualizar_estado_d0()

    def _on_dominio_acentuado(self, dominio):
        es_espacial = dominio == "Espacial"
        self.combo_acentuado.setEnabled(es_espacial)
        self._actualizar_estado_d0()

    def _actualizar_estado_d0(self):
        usa_frecuencia = (
            self.combo_dominio_suavizado.currentText() == "Frecuencial"
            or self.combo_dominio_acentuado.currentText() == "Frecuencial"
        )
        self.lbl_d0.setEnabled(usa_frecuencia)
        self.sl_d0.setEnabled(usa_frecuencia)

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
        d0 = getattr(self, "sl_d0", None)
        ruido_txt = f"{ruido.value()}%" if ruido else "0%"
        umbral_txt = str(umbral.value()) if umbral else "128"
        d0_txt = str(d0.value()) if d0 else "45"
        max_area_txt = str(max_area.value()) if max_area.value() > 0 else "sin límite"

        self.lbl_flujo.setText(
            "Activos: "
            f"R {ruido_txt} | "
            f"S {dominio_suavizado.currentText()}/{suavizado.currentText() if dominio_suavizado.currentText() == 'Espacial' else f'D0 {d0_txt}'} | "
            f"A {dominio_acentuado.currentText()}/{acentuado.currentText() if dominio_acentuado.currentText() == 'Espacial' else f'D0 {d0_txt}'} | "
            f"G {gradiente.currentText()} | "
            f"U {umbral_txt} | "
            f"Á {min_area.value()}-{max_area_txt}"
        )

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
            # Cargar esto limpio ayuda bastante para revisar las demas etapas.
            self.img_rgb = modelo.cargar_imagen(ruta)
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
            self.btn_procesar.setEnabled(True)
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
            self.ultimo_resultado = None
        except Exception as exc:
            self.lbl_estado.setText(f"Error al cargar la imagen: {exc}")

    def _limpiar_todo(self):
        if self.worker and self.worker.isRunning():
            return

        print("[ui] Limpiando resultados y reiniciando controles...")
        self.img_rgb = None
        self.ultimo_resultado = None
        self.lbl_path.setText("Ninguna imagen cargada")
        self.lbl_size.setText("")
        self.lbl_estado.setText("Carga una imagen para empezar.")
        self.sl_ruido.setValue(5)
        self.sl_mascara.setValue(3)
        self.sl_d0.setValue(45)
        self.sl_umbral.setValue(128)
        self.spin_min_area.setValue(50)
        self.spin_max_area.setValue(0)
        self.combo_dominio_suavizado.setCurrentText("Espacial")
        self.combo_suavizado.setCurrentText("Media")
        self.combo_dominio_acentuado.setCurrentText("Espacial")
        self.combo_acentuado.setCurrentText("Sobel")
        self.combo_gradiente.setCurrentText("Sobel")
        self._on_dominio_suavizado(self.combo_dominio_suavizado.currentText())
        self._on_dominio_acentuado(self.combo_dominio_acentuado.currentText())

        self.btn_procesar.setEnabled(False)
        self._mostrar_placeholders()
        self._actualizar_resumen_flujo()

    def _procesar_todo(self):
        if self.img_rgb is None:
            return

        print("[ui] Lanzando procesamiento desde la interfaz...")
        self.btn_procesar.setEnabled(False)
        self.btn_cargar.setEnabled(False)
        self.btn_limpiar.setEnabled(False)
        self.lbl_estado.setText("Aplicando el pipeline seleccionado...")
        parametros = ParametrosProcesamiento(
            img_rgb=self.img_rgb.copy(),
            ruido=self.sl_ruido.value() / 100.0,
            mascara=self._kernel_actual(),
            d0=self.sl_d0.value(),
            dominio_suavizado=self.combo_dominio_suavizado.currentText(),
            tipo_suavizado=self.combo_suavizado.currentText(),
            dominio_acentuado=self.combo_dominio_acentuado.currentText(),
            tipo_acentuado=self.combo_acentuado.currentText(),
            tipo_gradiente=self.combo_gradiente.currentText(),
            umbral=self.sl_umbral.value(),
            min_area=self.spin_min_area.value(),
            max_area=self.spin_max_area.value(),
        )

        self.worker = WorkerProcesoCompleto(parametros)
        self.worker.terminado.connect(self._on_proceso_listo)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_proceso_listo(self, resultado):
        print("[ui] Actualizando paneles con los resultados...")
        self.ultimo_resultado = resultado
        datos = resultado.a_diccionario()
        self.canvas_preprocesamiento.actualizar(datos)

        filtro = resultado.tipo_suavizado
        mascara = resultado.mascara

        self.canvas_suavizado.actualizar_titulos(
            [
                f"Ruido sal y pimienta ({resultado.ruido_porcentaje} %)",
                f"Suavizado {resultado.dominio_suavizado} / {filtro if resultado.dominio_suavizado == 'Espacial' else f'D0={resultado.d0}'}",
            ]
        )
        self.canvas_suavizado.actualizar(datos)
        self.canvas_acentuado.actualizar_titulos(
            [
                "Entrada suavizada",
                f"Acentuado {resultado.dominio_acentuado} / {resultado.tipo_acentuado if resultado.dominio_acentuado == 'Espacial' else f'D0={resultado.d0}'}",
            ]
        )
        self.canvas_acentuado.actualizar(datos)
        self.canvas_binarizacion.actualizar_titulos(
            [
                f"Acentuado {resultado.dominio_acentuado}",
                f"Binarizada (umbral={resultado.umbral})",
            ]
        )
        self.canvas_binarizacion.actualizar(datos)
        self.canvas_gradiente.actualizar_titulos(
            [
                f"Binarizada (umbral={resultado.umbral})",
                f"Gradiente {resultado.tipo_gradiente}",
                "Bordes binarios finales",
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
                f"Bordes binarios ({resultado.tipo_gradiente})",
                f"Bounding boxes ({n} regiones)",
            ]
        )
        self.canvas_regiones.actualizar(datos)
        self.canvas_recortes.actualizar_titulos(
            [
                f"Submatrices normalizadas — entrada al clasificador ({n} recortes)",
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

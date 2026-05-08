"""Ventana principal y armado general de la aplicación."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .controles import ControlesMixin
from .estilos import APP_STYLE
from .procesamiento import ProcesamientoMixin
from .tabs import TabsMixin


class VentanaPrincipal(ProcesamientoMixin, ControlesMixin, TabsMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesamiento de Imagenes: Dominio Espacial y Frecuencia")
        self.resize(1520, 920)
        self.setStyleSheet(APP_STYLE)

        self.img_rgb = None
        self.max_mascara = 3
        self.worker = None
        self.ultimo_resultado = None
        self.datos_parciales = {}

        self._build_ui()
        self._mostrar_placeholders()

    def _build_ui(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(374)

        sidebar_host_layout = QVBoxLayout(sidebar)
        sidebar_host_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_host_layout.setSpacing(0)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setObjectName("SidebarScrollArea")
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        sidebar_content = QWidget()
        sidebar_content.setFixedWidth(354)
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(20, 24, 18, 24)
        sidebar_layout.setSpacing(14)
        botones_layout = QHBoxLayout()
        botones_layout.setSpacing(8)

        self.btn_cargar = QPushButton("Cargar imagen")
        self.btn_cargar.setObjectName("SecondaryButton")
        self.btn_cargar.setMinimumWidth(0)
        self.btn_cargar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_cargar.setProperty("compact", True)
        self.btn_cargar.clicked.connect(self._cargar_imagen)
        botones_layout.addWidget(self.btn_cargar, 1)

        self.btn_limpiar = QPushButton("Limpiar")
        self.btn_limpiar.setObjectName("DangerButton")
        self.btn_limpiar.setMinimumWidth(0)
        self.btn_limpiar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_limpiar.setProperty("compact", True)
        self.btn_limpiar.clicked.connect(self._limpiar_todo)
        botones_layout.addWidget(self.btn_limpiar, 1)

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

        self.btn_guardar_resultados = QPushButton("Guardar final")
        self.btn_guardar_resultados.setObjectName("SecondaryButton")
        self.btn_guardar_resultados.setEnabled(False)
        self.btn_guardar_resultados.clicked.connect(self._guardar_resultados)
        sidebar_layout.addWidget(self.btn_guardar_resultados)

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
        self._on_dominio_suavizado(self.combo_dominio_suavizado.currentText())
        self._on_dominio_acentuado(self.combo_dominio_acentuado.currentText())
        self._actualizar_resumen_flujo()

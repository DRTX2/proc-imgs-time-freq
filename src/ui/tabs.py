"""Construcción de pestañas y áreas de resultados."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from .widgets import CanvasResultados


class TabsMixin:
    def _crear_tab_preprocesamiento(self):
        tab, layout = self._crear_tab_scrollable()

        self.canvas_preprocesamiento = CanvasResultados(
            [
                {"tipo": "imagen", "clave": "original",    "titulo": "Imagen original RGB"},
                {"tipo": "imagen", "clave": "gris",        "titulo": "Escala de grises"},
                {"tipo": "imagen", "clave": "ruido",       "titulo": "Con ruido sal y pimienta"},
                {"tipo": "imagen", "clave": "normalizada", "titulo": "Normalizada post-ruido"},
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
                {"tipo": "imagen", "clave": "normalizada", "titulo": "Base normalizada"},
                {"tipo": "imagen", "clave": "suavizada", "titulo": "Resultado suavizado"},
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
                {"tipo": "imagen", "clave": "suavizada", "titulo": "Base suavizada"},
                {"tipo": "imagen", "clave": "acentuada", "titulo": "Resultado acentuado"},
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
                {"tipo": "imagen", "clave": "gradiente_binario", "titulo": "Bordes binarios finales"},
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
                {"tipo": "imagen", "clave": "acentuada",         "titulo": "Base acentuada"},
                {"tipo": "imagen", "clave": "binaria_acentuada", "titulo": "Máscara binaria"},
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
                {"tipo": "imagen", "clave": "binaria", "titulo": "Máscara de detección"},
                {"tipo": "imagen", "clave": "bboxes",  "titulo": "Regiones detectadas"},
            ],
            1,
            2,
            self,
            altura=280,
            altura_maxima=420,
            expandible=True,
        )
        layout.addWidget(self.canvas_regiones)

        self.canvas_recortes = CanvasResultados(
            [
                {"tipo": "imagen", "clave": "recortes_tira", "titulo": "Submatrices normalizadas para clasificación"},
            ],
            1,
            1,
            self,
            altura=320,
            altura_maxima=520,
            expandible=True,
        )
        layout.addWidget(self.canvas_recortes)
        layout.addStretch(1)
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
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
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

import sys
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)
import modelo


# --- Utilidades ---

def np_a_pixmap(img_np):
    """Convierte un array NumPy (gris o RGB) a QPixmap."""
    if img_np.ndim == 2:
        img_np = np.ascontiguousarray(img_np)
        h, w = img_np.shape
        qimg = QImage(img_np.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())
    img_np = np.ascontiguousarray(img_np)
    h, w, _ = img_np.shape
    qimg = QImage(img_np.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def formato_resolucion(imagen):
    alto, ancho = imagen.shape[:2]
    return f"{ancho} x {alto} px"


# --- Worker para procesamiento en hilo secundario ---

class WorkerFiltro(QThread):
    terminado = Signal()

    def __init__(self, estado, dominio, params):
        super().__init__()
        self.estado = estado
        self.dominio = dominio
        self.params = params

    def run(self):
        if self.dominio == "espacial":
            self.estado.aplicar_filtro_espacial(**self.params)
        elif self.dominio == "frecuencia":
            self.estado.aplicar_filtro_frecuencia(**self.params)
        self.terminado.emit()


# --- Estilos ---

def estilo_slider(color):
    return f"""
    QSlider::groove:horizontal {{
        border: 1px solid #D7CCBD; height: 7px;
        background: #EDE4D9; border-radius: 4px;
    }}
    QSlider::sub-page:horizontal {{ background: {color}; border-radius: 4px; }}
    QSlider::add-page:horizontal {{ background: #E1D7CB; border-radius: 4px; }}
    QSlider::handle:horizontal {{
        background: #FFFDF8; border: 2px solid {color};
        width: 14px; margin: -5px 0; border-radius: 9px;
    }}
    """

APP_STYLE = """
QMainWindow { background: #F5EFE6; }
QWidget#AppRoot, QWidget#PageBg, QStackedWidget,
QStackedWidget > QWidget { background: #F5EFE6; }
QWidget { color: #3F3A36; font-size: 12px; }
QScrollArea { border: none; background: #F5EFE6; }
QFrame#Card {
    background: #FFFDF9; border: 1px solid #E0D5C7; border-radius: 14px;
}
QLabel#MainTitle { font-size: 26px; font-weight: 800; color: #3F3A36; }
QLabel#SectionTitle { font-size: 14px; font-weight: 800; color: #4A443F; }
QLabel#Muted { color: #7A6F64; font-size: 12px; }
QLabel#AccentGreen { color: #5F8A63; font-size: 14px; font-weight: 800; }
QPushButton {
    background: #5E5A56; color: white; border: none;
    border-radius: 8px; padding: 10px 14px; font-size: 13px; font-weight: 700;
}
QPushButton:hover { background: #4F4B47; }
QPushButton:disabled { background: #A09A94; }
QPushButton#PrimaryButton { background: #6E7B57; }
QPushButton#PrimaryButton:hover { background: #5F6D4A; }
QPushButton#NavButton {
    background: transparent; color: #7A6F64;
    border: 1px solid #DCCDBD; padding: 8px 16px;
}
QPushButton#NavButton[active="true"] {
    background: #FFFDF9; color: #4A443F; border: 1px solid #CDBCA9;
}
QComboBox {
    background: #FFFDF9; border: 1px solid #D7CCBD;
    border-radius: 6px; padding: 6px 10px; font-weight: 600;
}
QSpinBox {
    background: #FFFDF9; border: 1px solid #D7CCBD;
    border-radius: 6px; padding: 4px 8px;
}
"""


class VentanaPrincipal(QMainWindow):
    def __init__(self, estado):
        super().__init__()
        self.estado = estado
        self.worker = None
        self.setWindowTitle("Filtrado de Imágenes — Dominio Espacial y Frecuencia")
        self.resize(1400, 900)
        self.setStyleSheet(APP_STYLE)
        self.ui_ready = False
        self._build_ui()
        self.ui_ready = True
        self._refrescar_previews_base()

    # ---- Construcción de la UI ----

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("AppRoot")
        root_ly = QVBoxLayout(root)
        root_ly.setContentsMargins(18, 14, 18, 18)
        root_ly.setSpacing(12)

        titulo = QLabel("Filtrado de Imágenes")
        titulo.setObjectName("MainTitle")
        titulo.setAlignment(Qt.AlignCenter)
        root_ly.addWidget(titulo)

        # Navegación
        nav = QHBoxLayout()
        nav.addStretch(1)
        self.btn_nav = []
        for i, txt in enumerate(["Imagen y Ruido", "Dominio Espacial", "Dominio Frecuencia"]):
            b = QPushButton(txt)
            b.setObjectName("NavButton")
            b.clicked.connect(lambda _, idx=i: self._ir_pagina(idx))
            nav.addWidget(b)
            self.btn_nav.append(b)
        nav.addStretch(1)
        root_ly.addLayout(nav)

        # Páginas
        self.stacked = QStackedWidget()
        self.stacked.addWidget(self._wrap_scroll(self._pagina_ruido()))
        self.stacked.addWidget(self._wrap_scroll(self._pagina_espacial()))
        self.stacked.addWidget(self._wrap_scroll(self._pagina_frecuencia()))
        root_ly.addWidget(self.stacked, 1)

        self.stacked.currentChanged.connect(self._actualizar_nav)
        self._actualizar_nav(0)
        self.setCentralWidget(root)

    def _wrap_scroll(self, w):
        s = QScrollArea()
        s.setWidgetResizable(True)
        s.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        s.setWidget(w)
        return s

    def _card(self):
        f = QFrame()
        f.setObjectName("Card")
        return f

    def _preview(self, alto=180):
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setMinimumHeight(alto)
        lbl.setMaximumHeight(alto)
        lbl.setStyleSheet(
            "background: #FBF7F0; border: 1px solid #E3D9CD;"
            "border-radius: 10px; padding: 6px;"
        )
        return lbl

    def _set_img(self, label, img_np):
        sz = label.size()
        if sz.width() <= 10 or sz.height() <= 10:
            h = label.minimumHeight() or 180
            sz = QSize(int(h * 1.5), h)
        label.setPixmap(
            np_a_pixmap(img_np).scaled(sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ---- Página 1: Carga de imagen y ruido ----

    def _pagina_ruido(self):
        page = QWidget()
        page.setObjectName("PageBg")
        ly = QVBoxLayout(page)
        ly.setContentsMargins(6, 8, 6, 16)
        ly.setSpacing(14)

        ly.addWidget(QLabel(
            "Carga de Imagen y Ruido Sal & Pimienta",
            alignment=Qt.AlignCenter, objectName="SectionTitle",
            styleSheet="font-size: 18px; font-weight: 900;"
        ))

        # Previews: original RGB, gris, con ruido
        grid = QGridLayout()
        grid.setSpacing(12)
        ly.addLayout(grid)

        self.prev_rgb = self._preview(200)
        self.prev_gris = self._preview(200)
        self.prev_ruido = self._preview(200)

        for col, (titulo, prev) in enumerate([
            ("Imagen RGB Original", self.prev_rgb),
            ("Escala de Grises (Luma)", self.prev_gris),
            ("Con Ruido Sal y Pimienta", self.prev_ruido),
        ]):
            card = self._card()
            cly = QVBoxLayout(card)
            cly.setContentsMargins(12, 12, 12, 12)
            cly.setSpacing(8)
            cly.addWidget(QLabel(titulo, alignment=Qt.AlignCenter, objectName="SectionTitle"))
            cly.addWidget(prev)
            grid.addWidget(card, 0, col)
            grid.setColumnStretch(col, 1)

        # Controles
        ctrl = self._card()
        cly = QVBoxLayout(ctrl)
        cly.setContentsMargins(16, 16, 16, 16)
        cly.setSpacing(12)

        # Botón cargar imagen
        btn_cargar = QPushButton("Cargar Imagen RGB")
        btn_cargar.setObjectName("PrimaryButton")
        btn_cargar.clicked.connect(self._seleccionar_imagen)
        cly.addWidget(btn_cargar)

        self.lbl_path = QLabel("Usando lienzo vacío.", objectName="Muted", wordWrap=True)
        cly.addWidget(self.lbl_path)

        # Slider probabilidad de ruido
        row = QHBoxLayout()
        row.addWidget(QLabel(
            "Probabilidad de ruido:",
            styleSheet="font-weight: 800; font-size: 13px;"
        ))
        self.lbl_prob = QLabel("5%", objectName="Muted", styleSheet="font-weight: 700;")
        row.addWidget(self.lbl_prob)
        row.addStretch(1)
        cly.addLayout(row)

        self.slider_ruido = QSlider(Qt.Horizontal)
        self.slider_ruido.setRange(1, 50)
        self.slider_ruido.setValue(5)
        self.slider_ruido.setStyleSheet(estilo_slider("#8B7B67"))
        self.slider_ruido.valueChanged.connect(self._on_ruido_changed)
        cly.addWidget(self.slider_ruido)

        cly.addWidget(QLabel(
            "Controla el porcentaje total de píxeles afectados "
            "(mitad sal/blanco, mitad pimienta/negro).",
            objectName="Muted"
        ))

        btn_aplicar_ruido = QPushButton("Aplicar Ruido")
        btn_aplicar_ruido.clicked.connect(self._aplicar_ruido)
        cly.addWidget(btn_aplicar_ruido)

        ly.addWidget(ctrl)
        return page

    # ---- Página 2: Dominio espacial ----

    def _pagina_espacial(self):
        page = QWidget()
        page.setObjectName("PageBg")
        ly = QVBoxLayout(page)
        ly.setContentsMargins(6, 8, 6, 16)
        ly.setSpacing(14)

        ly.addWidget(QLabel(
            "Filtrado en Dominio Espacial",
            alignment=Qt.AlignCenter, objectName="SectionTitle",
            styleSheet="font-size: 18px; font-weight: 900;"
        ))

        # Previews
        grid = QGridLayout()
        grid.setSpacing(12)
        ly.addLayout(grid)

        self.prev_esp_entrada = self._preview(200)
        self.prev_esp_resultado = self._preview(200)
        self.prev_esp_binario = self._preview(200)

        for col, (titulo, prev) in enumerate([
            ("Imagen con Ruido (entrada)", self.prev_esp_entrada),
            ("Resultado del Filtro", self.prev_esp_resultado),
            ("Binarización (umbral = media)", self.prev_esp_binario),
        ]):
            card = self._card()
            cly = QVBoxLayout(card)
            cly.setContentsMargins(12, 12, 12, 12)
            cly.setSpacing(8)
            cly.addWidget(QLabel(titulo, alignment=Qt.AlignCenter, objectName="SectionTitle"))
            cly.addWidget(prev)
            grid.addWidget(card, 0, col)
            grid.setColumnStretch(col, 1)

        # Controles
        ctrl = self._card()
        cly = QVBoxLayout(ctrl)
        cly.setContentsMargins(16, 16, 16, 16)
        cly.setSpacing(12)

        cly.addWidget(QLabel("Controles", objectName="SectionTitle"))

        # Selector de filtro
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Tipo de filtro:", styleSheet="font-weight: 800;"))
        self.combo_filtro = QComboBox()
        self.combo_filtro.addItems(["Media", "Mediana", "Moda"])
        row1.addWidget(self.combo_filtro)
        row1.addStretch(1)
        cly.addLayout(row1)

        # Tamaño del kernel
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Tamaño del kernel:", styleSheet="font-weight: 800;"))
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(3, 15)
        self.spin_kernel.setSingleStep(2)
        self.spin_kernel.setValue(3)
        row2.addWidget(self.spin_kernel)
        row2.addStretch(1)
        cly.addLayout(row2)

        self.lbl_info_espacial = QLabel("", objectName="Muted")
        cly.addWidget(self.lbl_info_espacial)

        self.btn_aplicar_esp = QPushButton("Aplicar Filtro Espacial")
        self.btn_aplicar_esp.setObjectName("PrimaryButton")
        self.btn_aplicar_esp.clicked.connect(self._aplicar_espacial)
        cly.addWidget(self.btn_aplicar_esp)

        self.lbl_estado_esp = QLabel("", objectName="Muted")
        cly.addWidget(self.lbl_estado_esp)

        ly.addWidget(ctrl)
        return page

    # ---- Página 3: Dominio de frecuencia ----

    def _pagina_frecuencia(self):
        page = QWidget()
        page.setObjectName("PageBg")
        ly = QVBoxLayout(page)
        ly.setContentsMargins(6, 8, 6, 16)
        ly.setSpacing(14)

        ly.addWidget(QLabel(
            "Filtrado en Dominio de Frecuencia (FFT)",
            alignment=Qt.AlignCenter, objectName="SectionTitle",
            styleSheet="font-size: 18px; font-weight: 900;"
        ))

        grid = QGridLayout()
        grid.setSpacing(12)
        ly.addLayout(grid)

        self.prev_freq_entrada = self._preview(200)
        self.prev_freq_espectro = self._preview(200)
        self.prev_freq_resultado = self._preview(200)

        for col, (titulo, prev) in enumerate([
            ("Imagen con Ruido (entrada)", self.prev_freq_entrada),
            ("Espectro de Magnitud (FFT)", self.prev_freq_espectro),
            ("Resultado (FFT Inversa)", self.prev_freq_resultado),
        ]):
            card = self._card()
            cly = QVBoxLayout(card)
            cly.setContentsMargins(12, 12, 12, 12)
            cly.setSpacing(8)
            cly.addWidget(QLabel(titulo, alignment=Qt.AlignCenter, objectName="SectionTitle"))
            cly.addWidget(prev)
            grid.addWidget(card, 0, col)
            grid.setColumnStretch(col, 1)

        # Controles
        ctrl = self._card()
        cly = QVBoxLayout(ctrl)
        cly.setContentsMargins(16, 16, 16, 16)
        cly.setSpacing(12)

        cly.addWidget(QLabel("Controles", objectName="SectionTitle"))

        row = QHBoxLayout()
        row.addWidget(QLabel("Radio del filtro pasabajas:", styleSheet="font-weight: 800;"))
        self.lbl_radio = QLabel("30", objectName="Muted", styleSheet="font-weight: 700;")
        row.addWidget(self.lbl_radio)
        row.addStretch(1)
        cly.addLayout(row)

        self.slider_radio = QSlider(Qt.Horizontal)
        self.slider_radio.setRange(1, 200)
        self.slider_radio.setValue(30)
        self.slider_radio.setStyleSheet(estilo_slider("#6E789D"))
        self.slider_radio.valueChanged.connect(
            lambda v: self.lbl_radio.setText(str(v))
        )
        cly.addWidget(self.slider_radio)

        cly.addWidget(QLabel(
            "Un radio pequeño deja pasar solo frecuencias bajas (suaviza mucho). "
            "Un radio grande conserva más detalle.",
            objectName="Muted"
        ))

        self.btn_aplicar_freq = QPushButton("Aplicar Filtro en Frecuencia")
        self.btn_aplicar_freq.setObjectName("PrimaryButton")
        self.btn_aplicar_freq.clicked.connect(self._aplicar_frecuencia)
        cly.addWidget(self.btn_aplicar_freq)

        self.lbl_estado_freq = QLabel("", objectName="Muted")
        cly.addWidget(self.lbl_estado_freq)

        ly.addWidget(ctrl)
        return page

    # ---- Navegación ----

    def _ir_pagina(self, idx):
        self.stacked.setCurrentIndex(idx)

    def _actualizar_nav(self, idx):
        for i, b in enumerate(self.btn_nav):
            b.setProperty("active", i == idx)
            b.style().unpolish(b)
            b.style().polish(b)
            b.update()

    # ---- Callbacks ----

    def _seleccionar_imagen(self):
        archivo, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen RGB", str(Path.cwd()),
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if archivo:
            self.estado.establecer_imagen(archivo)
            self.lbl_path.setText(f"Imagen: {Path(archivo).name}")
            self._refrescar_previews_base()

    def _on_ruido_changed(self, valor):
        self.lbl_prob.setText(f"{valor}%")

    def _aplicar_ruido(self):
        prob = self.slider_ruido.value() / 100.0
        self.estado.aplicar_ruido(prob)
        self._refrescar_previews_base()

    def _aplicar_espacial(self):
        tipo_map = {0: "media", 1: "mediana", 2: "moda"}
        tipo = tipo_map[self.combo_filtro.currentIndex()]
        tam = self.spin_kernel.value()
        # Forzar que sea impar
        if tam % 2 == 0:
            tam += 1
            self.spin_kernel.setValue(tam)

        self.btn_aplicar_esp.setEnabled(False)
        self.lbl_estado_esp.setText("Procesando... (puede tardar unos segundos)")

        self.worker = WorkerFiltro(self.estado, "espacial", {
            "tipo": tipo, "tam_kernel": tam
        })
        self.worker.terminado.connect(self._espacial_listo)
        self.worker.start()

    def _espacial_listo(self):
        self.btn_aplicar_esp.setEnabled(True)
        media = modelo.calcular_media(self.resultado_espacial_ref())
        self.lbl_estado_esp.setText(f"Listo. Umbral de binarización (media): {media:.1f}")
        self.lbl_info_espacial.setText(
            f"Filtro: {self.estado.tipo_filtro_espacial} | "
            f"Kernel: {self.estado.tam_kernel}x{self.estado.tam_kernel}"
        )
        self._set_img(self.prev_esp_resultado, self.estado.resultado_espacial)
        self._set_img(self.prev_esp_binario, self.estado.resultado_binario)

    def resultado_espacial_ref(self):
        return self.estado.resultado_espacial

    def _aplicar_frecuencia(self):
        radio = self.slider_radio.value()

        self.btn_aplicar_freq.setEnabled(False)
        self.lbl_estado_freq.setText("Calculando FFT... (puede tardar unos segundos)")

        self.worker = WorkerFiltro(self.estado, "frecuencia", {"radio": radio})
        self.worker.terminado.connect(self._frecuencia_listo)
        self.worker.start()

    def _frecuencia_listo(self):
        self.btn_aplicar_freq.setEnabled(True)
        self.lbl_estado_freq.setText(
            f"Listo. Radio: {self.estado.radio_frecuencia} px"
        )
        self._set_img(self.prev_freq_espectro, self.estado.espectro_magnitud)
        self._set_img(self.prev_freq_resultado, self.estado.resultado_frecuencia)

    def _refrescar_previews_base(self):
        """Actualiza las previews de la página de ruido y las entradas de las otras."""
        if not self.ui_ready:
            return
        self._set_img(self.prev_rgb, self.estado.img_rgb)
        self._set_img(self.prev_gris, self.estado.img_gris)
        self._set_img(self.prev_ruido, self.estado.img_ruido)
        self._set_img(self.prev_esp_entrada, self.estado.img_ruido)
        self._set_img(self.prev_freq_entrada, self.estado.img_ruido)

    def closeEvent(self, event):
        """Detener los threads cuando se cierra la ventana."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
        super().closeEvent(event)

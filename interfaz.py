import sys
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget, QTabWidget,
    QGridLayout, QComboBox
)
import modelo


# --- Utilidades ---

def to_rgb_format(img_np):
    """Convierte o asegura que la imagen tenga 3 canales para presentar en RGB."""
    if img_np.ndim == 2:
        return np.stack((img_np, img_np, img_np), axis=2)
    return img_np

def np_to_pixmap(img_np):
    """Convierte array numpy a QPixmap (asegurando formato RGB)"""
    img_rgb = to_rgb_format(img_np)
    img_rgb = np.ascontiguousarray(img_rgb)
    h, w, _ = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

def estilo_slider(color):
    return f"""
    QSlider::groove:horizontal {{
        border: 1px solid #D7CCBD; height: 7px; background: #EDE4D9; border-radius: 4px;
    }}
    QSlider::sub-page:horizontal {{ background: {color}; border-radius: 4px; }}
    QSlider::add-page:horizontal {{ background: #E1D7CB; border-radius: 4px; }}
    QSlider::handle:horizontal {{
        background: #FFFDF8; border: 2px solid {color}; width: 14px; margin: -5px 0; border-radius: 9px;
    }}
    """

APP_STYLE = """
QMainWindow { background: #F5EFE6; }
QWidget { color: #3F3A36; font-size: 12px; }
QTabWidget::pane { border: 1px solid #E0D5C7; background: #FFFDF9; border-radius: 8px; }
QTabBar::tab { background: #EAE2D6; border: 1px solid #D7CCBD; padding: 8px 16px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; color: #5E5A56; font-weight: bold; }
QTabBar::tab:selected { background: #FFFDF9; border-bottom-color: #FFFDF9; color: #2B2825; }
QFrame#Card { background: #FFFDF9; border: 1px solid #E0D5C7; border-radius: 14px; }
QLabel#MainTitle { font-size: 24px; font-weight: 800; color: #3F3A36; }
QLabel#Muted { color: #7A6F64; font-size: 12px; }
QPushButton { background: #5E5A56; color: white; border: none; border-radius: 8px; padding: 10px 14px; font-size: 13px; font-weight: 700; }
QPushButton:hover { background: #4F4B47; }
QPushButton:disabled { background: #A09A94; }
QPushButton#PrimaryButton { background: #6E7B57; }
QPushButton#PrimaryButton:hover { background: #5F6D4A; }
"""


class WorkerProcesoCompleto(QThread):
    terminado = Signal(dict)
    error = Signal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        try:
            res = {}
            img_rgb = self.args["img_rgb"]
            t_thresh = self.args["thresh"]
            t_ruido = self.args["ruido"]
            tam_masc = self.args["mascara"]
            sigma = self.args["sigma"]

            # 1. Preprocesamiento (Background task)
            img_norm_rgb = modelo.normalizar_histograma_rgb(img_rgb)
            img_gris = modelo.convertir_a_grises(img_norm_rgb)
            img_bin = modelo.binarizar_imagen(img_gris, t_thresh)

            # 2. Ruido a la imagen en grises preprocesada
            img_ruido = modelo.agregar_ruido_sal_pimienta(img_gris, t_ruido)

            # 3. Dominio Espacial (Filtros atenuan ruido)
            tipo_filtro = self.args["tipo_filtro"]
            if tipo_filtro == "Media":
                f_espacial = modelo.filtro_media(img_ruido, tam_masc)
            elif tipo_filtro == "Mediana":
                f_espacial = modelo.filtro_mediana(img_ruido, tam_masc)
            else:
                f_espacial = modelo.filtro_moda(img_ruido, tam_masc)

            # 4. Dominio Frecuencia (Filtros atenuan ruido)
            r_freq, e_orig, e_filt, masc_gauss = modelo.fourier_procesada_completa(img_ruido, sigma)

            res = {
                "norm_rgb": img_norm_rgb,
                "gris": img_gris,
                "binaria": img_bin,
                "ruido": img_ruido,
                "f_espacial": f_espacial,
                "tipo_filtro": tipo_filtro,
                "resultado_freq": r_freq,
                "esp_orig": modelo.espectro_log(e_orig),
                "esp_filt": modelo.espectro_log(e_filt),
                "masc_gauss": masc_gauss
            }
            self.terminado.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, layout_type="prep"):
        self.fig = Figure(figsize=(8, 6), facecolor="#FFFDF9")
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.15, hspace=0.25)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.axes = []
        if layout_type == "prep":
            self.titulos_base = ["Original RGB", "Normalizada RGB", "Escala de Grises", "Binarizada"]
        elif layout_type == "espacial":
            self.titulos_base = ["Grises (Base)", "Con Ruido S&P", "Resultado Filtro Espacial", ""]
        else:
            self.titulos_base = ["Espectro Original", "Máscara Gaussiana", "Espectro Filtrado", "Resultado Frecuencia"]

        for i in range(4):
            ax = self.fig.add_subplot(2, 2, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            self.axes.append(ax)
        
        self.titulos_actuales = list(self.titulos_base)

    def actualizar_imagenes(self, imgs, cmaps=None, titulos_override=None):
        """Actualiza los 4 subplots a la vez."""
        if cmaps is None:
            cmaps = [None] * 4
        
        if titulos_override:
            for i, t in enumerate(titulos_override):
                if t is not None:
                    self.titulos_actuales[i] = t

        for i, (ax, img, cmap) in enumerate(zip(self.axes, imgs, cmaps)):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Aplicar el título almacenado
            ax.set_title(self.titulos_actuales[i], fontsize=11, fontweight="bold", color="#4A443F")

            if cmap:
                ax.imshow(img, cmap=cmap)
            else:
                img_show = to_rgb_format(img)
                ax.imshow(img_show)
        self.draw_idle()


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesamiento de Imágenes")
        self.resize(1300, 850)
        self.setStyleSheet(APP_STYLE)
        
        self.img_rgb = None
        self.max_mascara = 3
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        root_ly = QVBoxLayout(root)
        root_ly.setContentsMargins(16, 16, 16, 16)
        root_ly.setSpacing(12)

        header = QLabel("Filtrado de Imágenes: Dominio Tiempo y Frecuencia")
        header.setObjectName("MainTitle")
        header.setAlignment(Qt.AlignCenter)
        root_ly.addWidget(header)

        # Panel Principal: Controles (Izquierda) + Tabs (Derecha)
        h_split = QHBoxLayout()
        h_split.setSpacing(16)

        # ---- Panel de Controles (Izquierda) ----
        ctrl_card = QFrame()
        ctrl_card.setObjectName("Card")
        ctrl_card.setFixedWidth(320)
        ctrl_ly = QVBoxLayout(ctrl_card)
        ctrl_ly.setContentsMargins(16, 20, 16, 20)
        ctrl_ly.setSpacing(18)

        # 1. Cargar
        btn_cargar = QPushButton("Cargar Imagen")
        btn_cargar.setObjectName("PrimaryButton")
        btn_cargar.clicked.connect(self._cargar_imagen)
        ctrl_ly.addWidget(btn_cargar)
        
        self.lbl_path = QLabel("Ninguna imagen cargada", objectName="Muted")
        self.lbl_path.setWordWrap(True)
        ctrl_ly.addWidget(self.lbl_path)
        
        ctrl_ly.addWidget(QFrame(frameShape=QFrame.HLine, frameShadow=QFrame.Sunken))

        # 2. Sliders
        # Ruido
        r_ly = QVBoxLayout()
        r_ly.addWidget(QLabel("Ruido S&P (%):", styleSheet="font-weight: bold;"))
        self.sl_ruido = QSlider(Qt.Horizontal)
        self.sl_ruido.setRange(0, 100)
        self.sl_ruido.setValue(5)
        self.sl_ruido.setStyleSheet(estilo_slider("#B46A6A"))
        self.lbl_ruido = QLabel("5%")
        self.sl_ruido.valueChanged.connect(lambda v: self.lbl_ruido.setText(f"{v}%"))
        r_row = QHBoxLayout()
        r_row.addWidget(self.sl_ruido)
        r_row.addWidget(self.lbl_ruido)
        r_ly.addLayout(r_row)
        ctrl_ly.addLayout(r_ly)

        # Threshold
        t_ly = QVBoxLayout()
        t_ly.addWidget(QLabel("Threshold Binarización:", styleSheet="font-weight: bold;"))
        self.sl_thresh = QSlider(Qt.Horizontal)
        self.sl_thresh.setRange(0, 255)
        self.sl_thresh.setValue(127)
        self.sl_thresh.setStyleSheet(estilo_slider("#5F8A63"))
        self.lbl_thresh = QLabel("127")
        self.sl_thresh.valueChanged.connect(lambda v: self.lbl_thresh.setText(str(v)))
        t_row = QHBoxLayout()
        t_row.addWidget(self.sl_thresh)
        t_row.addWidget(self.lbl_thresh)
        t_ly.addLayout(t_row)
        ctrl_ly.addLayout(t_ly)

        # Máscara Tiempo
        m_ly = QVBoxLayout()
        m_ly.addWidget(QLabel("Tamaño Máscara (Tiempo):", styleSheet="font-weight: bold;"))
        self.sl_mascara = QSlider(Qt.Horizontal)
        self.sl_mascara.setRange(3, 3)
        self.sl_mascara.setSingleStep(2)
        self.sl_mascara.setValue(3)
        self.sl_mascara.setStyleSheet(estilo_slider("#6E789D"))
        self.lbl_mascara = QLabel("3x3")
        self.sl_mascara.valueChanged.connect(self._on_mascara)
        m_row = QHBoxLayout()
        m_row.addWidget(self.sl_mascara)
        m_row.addWidget(self.lbl_mascara)
        m_ly.addLayout(m_row)
        ctrl_ly.addLayout(m_ly)

        # Sigma Frecuencia
        s_ly = QVBoxLayout()
        s_ly.addWidget(QLabel("Sigma Máscara (Frecuencia):", styleSheet="font-weight: bold;"))
        self.sl_sigma = QSlider(Qt.Horizontal)
        self.sl_sigma.setRange(1, 100)
        self.sl_sigma.setValue(30)
        self.sl_sigma.setStyleSheet(estilo_slider("#8A7F73"))
        self.lbl_sigma = QLabel("30")
        self.sl_sigma.valueChanged.connect(lambda v: self.lbl_sigma.setText(str(v)))
        s_row = QHBoxLayout()
        s_row.addWidget(self.sl_sigma)
        s_row.addWidget(self.lbl_sigma)
        s_ly.addLayout(s_row)
        ctrl_ly.addLayout(s_ly)

        # Tipo de Filtro Espacial
        f_ly = QVBoxLayout()
        f_ly.addWidget(QLabel("Tipo de Filtro Espacial:", styleSheet="font-weight: bold;"))
        self.combo_filtro = QComboBox()
        self.combo_filtro.addItems(["Media", "Mediana", "Moda"])
        self.combo_filtro.setStyleSheet("background: #FFFDF9; border: 1px solid #D7CCBD; padding: 4px; border-radius: 4px;")
        f_ly.addWidget(self.combo_filtro)
        ctrl_ly.addLayout(f_ly)

        ctrl_ly.addStretch(1)

        # 3. Acciones
        self.btn_procesar = QPushButton("Aplicar Cambios y Procesar")
        self.btn_procesar.clicked.connect(self._procesar_todo)
        ctrl_ly.addWidget(self.btn_procesar)
        
        self.lbl_status = QLabel("Esperando imagen...", objectName="Muted")
        self.lbl_status.setWordWrap(True)
        ctrl_ly.addWidget(self.lbl_status)

        h_split.addWidget(ctrl_card)

        # ---- Panel de Pestañas (Derecha) ----
        self.tabs = QTabWidget()
        
        # Tab 1: Preprocesamiento
        self.canvas_prep = MatplotlibCanvas(self, "prep")
        self.tabs.addTab(self.canvas_prep, "1. Preprocesamiento")
        
        # Tab 2: Dominio Espacial
        self.canvas_esp = MatplotlibCanvas(self, "espacial")
        self.tabs.addTab(self.canvas_esp, "2. Dominio Espacial (Filtros)")
        
        # Tab 3: Dominio Frecuencia
        self.canvas_freq = MatplotlibCanvas(self, "frecuencia")
        self.tabs.addTab(self.canvas_freq, "3. Dominio Frecuencia (Filtros)")

        h_split.addWidget(self.tabs, 1)

        root_ly.addLayout(h_split)
        self.setCentralWidget(root)

    def _on_mascara(self, val):
        if val % 2 == 0:
            val += 1
            self.sl_mascara.setValue(val)
            return
        self.lbl_mascara.setText(f"{val}x{val}")

    def _cargar_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", str(Path.cwd()), "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if ruta:
            try:
                self.img_rgb = modelo.cargar_imagen(ruta)
                self.lbl_path.setText(Path(ruta).name)
                
                # Pre-calcular el tamaño máximo de la máscara
                gris = modelo.convertir_a_grises(self.img_rgb)
                self.max_mascara = modelo.calcular_tamano_mascara_maximo(gris)
                self.sl_mascara.setRange(3, self.max_mascara)
                
                # Procesar automáticamente con valores por defecto
                self._procesar_todo()
            except Exception as e:
                self.lbl_status.setText(f"Error: {e}")

    def _procesar_todo(self):
        if self.img_rgb is None: 
            return
            
        self.btn_procesar.setEnabled(False)
        self.lbl_status.setText("Procesando pipeline completo (Background)...")
        
        args = {
            "img_rgb": self.img_rgb.copy(),
            "thresh": self.sl_thresh.value(),
            "ruido": self.sl_ruido.value() / 100.0,
            "mascara": self.sl_mascara.value(),
            "sigma": self.sl_sigma.value(),
            "tipo_filtro": self.combo_filtro.currentText()
        }
        
        self.worker = WorkerProcesoCompleto(args)
        self.worker.terminado.connect(self._on_proceso_listo)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_proceso_listo(self, res):
        # Actualizar gráficas Tab 1: Preprocesamiento
        self.canvas_prep.actualizar_imagenes(
            [self.img_rgb, res["norm_rgb"], res["gris"], res["binaria"]]
        )
        
        # Actualizar gráficas Tab 2: Dominio Espacial
        # Pasamos el nombre del filtro en el override de títulos
        self.canvas_esp.actualizar_imagenes(
            [res["gris"], res["ruido"], res["f_espacial"], np.zeros_like(res["gris"])],
            cmaps=["gray", "gray", "gray", "gray"],
            titulos_override=[None, None, f"Filtro: {res['tipo_filtro']}", ""]
        )
        
        # Actualizar gráficas Tab 3: Dominio Frecuencia
        # Los espectros se ven mejor con colormap gray, la máscara con inferno.
        # El resultado se manda a RGB vía `to_rgb_format` desde el canvas si no se le pasa cmap.
        self.canvas_freq.actualizar_imagenes(
            [res["esp_orig"], res["masc_gauss"], res["esp_filt"], res["resultado_freq"]],
            cmaps=["gray", "inferno", "gray", None]
        )
        
        self.btn_procesar.setEnabled(True)
        self.lbl_status.setText("Pipeline procesado con éxito. Todo en formato RGB para visualización.")

    def _on_error(self, msg):
        self.lbl_status.setText(f"Error: {msg}")
        self.btn_procesar.setEnabled(True)

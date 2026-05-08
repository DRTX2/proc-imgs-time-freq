"""Widgets auxiliares usados por la ventana principal."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QComboBox, QSizePolicy

from src.aplicacion import ProcesadorImagen

def to_rgb_format(img_np):
    """Asegura que la imagen se muestre como RGB en matplotlib."""
    if img_np.ndim == 2:
        return np.stack((img_np, img_np, img_np), axis=2)
    return img_np

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


class WorkerProcesoEtapa(QThread):
    terminado = Signal(str, object)
    error = Signal(str)

    def __init__(self, parametros, etapa, procesador=None):
        super().__init__()
        self.parametros = parametros
        self.etapa = etapa
        self.procesador = procesador or ProcesadorImagen()

    def run(self):
        try:
            datos = self.procesador.ejecutar_hasta(self.parametros, self.etapa)
            self.terminado.emit(self.etapa, datos)
        except Exception as exc:
            print(f"[pipeline] Error: {exc}")
            self.error.emit(str(exc))


class ComboSoloDropdown(QComboBox):
    """Evita cambios accidentales con la rueda del mouse."""

    def wheelEvent(self, event):
        event.ignore()


class CanvasResultados(FigureCanvas):
    def __init__(self, items, filas, columnas, parent=None, altura=None, altura_maxima=None, expandible=False):
        altura = altura or self._calcular_altura_canvas(len(items), filas, columnas)
        self.fig = Figure(figsize=(12, altura / 100.0), facecolor="white")
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.10, wspace=0.18, hspace=0.24)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background: white;")
        self.setMinimumHeight(altura)
        if expandible:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            if altura_maxima is not None:
                self.setMaximumHeight(altura_maxima)
        else:
            self.setMaximumHeight(altura if altura_maxima is None else altura_maxima)

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

"""Exportación de resultados finales del pipeline."""

from pathlib import Path

import cv2
import numpy as np


def _a_bgr_para_png(imagen):
    """Convierte RGB a BGR solo al momento de escribir con OpenCV."""
    if imagen.ndim == 2:
        return imagen

    alto, ancho, canales = imagen.shape
    if canales < 3:
        return imagen

    salida = np.zeros((alto, ancho, 3), dtype=np.uint8)
    for fila in range(alto):
        for columna in range(ancho):
            salida[fila, columna, 0] = imagen[fila, columna, 2]
            salida[fila, columna, 1] = imagen[fila, columna, 1]
            salida[fila, columna, 2] = imagen[fila, columna, 0]
    return salida


def guardar_imagen_png(imagen, ruta):
    """Guarda arreglos 2D/RGB como PNG sin depender del formato de la ruta."""
    if imagen is None:
        return False

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    imagen_salida = _a_bgr_para_png(imagen)
    ok, buffer = cv2.imencode(".png", imagen_salida)
    if not ok:
        raise ValueError(f"No se pudo codificar la imagen: {ruta.name}")
    buffer.tofile(str(ruta))
    return True


def guardar_metadatos_recortes(recortes, ruta):
    """Deja un índice pequeño para rastrear cada crop con su bounding box."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open("w", encoding="utf-8") as archivo:
        archivo.write("indice,id_region,area,perimetro,fila_min,col_min,fila_max,col_max,archivo\n")
        for recorte in recortes:
            f0, c0, f1, c1 = recorte["bbox"]
            nombre = f"recorte_{recorte['indice']:03d}_region_{recorte['id_region']}.png"
            archivo.write(
                f"{recorte['indice']},{recorte['id_region']},{recorte['area']},"
                f"{recorte['perimetro']},{f0},{c0},{f1},{c1},{nombre}\n"
            )


def guardar_resultados_finales(resultado, carpeta):
    """
    Guarda las salidas que sirven para defender el proceso y preparar clasificación.
    Devuelve la lista de archivos generados.
    """
    datos = resultado.a_diccionario() if hasattr(resultado, "a_diccionario") else resultado
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    archivos = []
    imagenes = [
        ("01_gradiente.png", datos.get("gradiente_final")),
        ("02_gradiente_binario.png", datos.get("gradiente_binario")),
        ("03_mascara_regiones.png", datos.get("binaria")),
        ("04_bounding_boxes.png", datos.get("bboxes")),
        ("05_recortes_mosaico.png", datos.get("recortes_tira")),
    ]

    for nombre, imagen in imagenes:
        ruta = carpeta / nombre
        if guardar_imagen_png(imagen, ruta):
            archivos.append(ruta)

    recortes = datos.get("recortes_clasificador") or []
    carpeta_recortes = carpeta / "recortes_clasificador"
    for recorte in recortes:
        nombre = f"recorte_{recorte['indice']:03d}_region_{recorte['id_region']}.png"
        ruta = carpeta_recortes / nombre
        if guardar_imagen_png(recorte["imagen"], ruta):
            archivos.append(ruta)

    if recortes:
        ruta_metadatos = carpeta_recortes / "metadata_recortes.csv"
        guardar_metadatos_recortes(recortes, ruta_metadatos)
        archivos.append(ruta_metadatos)

    return archivos

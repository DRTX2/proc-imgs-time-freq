"""Binarización, componentes conexas y recortes por bounding box."""

import numpy as np

def binarizar_imagen(imagen_gris, umbral):
    """
    Binarización manual por umbral global.
    Pixel >= umbral  =>  255 (blanco / objeto)
    Pixel <  umbral  =>  0   (negro / fondo)
    Ejemplo corto: con umbral 128, un píxel 170 queda blanco y uno 90 queda negro.
    """
    print(f"[modelo] Binarizando con umbral={umbral}...")
    alto, ancho = imagen_gris.shape
    resultado = np.zeros((alto, ancho), dtype=np.uint8)
    for fila in range(alto):
        for columna in range(ancho):
            if int(imagen_gris[fila, columna]) >= umbral:
                resultado[fila, columna] = 255
    return resultado


def etiquetar_regiones_bfs(imagen_binaria, min_area=50, max_area=None):
    """
    Etiqueta regiones conexas (4-vecindad) con BFS manual.
    Devuelve lista de regiones ordenadas de mayor a menor area.
    Ejemplo corto: tres píxeles blancos tocándose por arriba/abajo/izquierda/derecha
    forman una sola región; si están separados, nacen regiones distintas.
    """
    print(f"[modelo] Etiquetando regiones BFS (min_area={min_area}, max_area={max_area})...")
    alto, ancho = imagen_binaria.shape
    visitado = [[False] * ancho for _ in range(alto)]
    regiones = []
    label = 0
    vecinos_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for fila_ini in range(alto):
        for col_ini in range(ancho):
            if imagen_binaria[fila_ini, col_ini] == 255 and not visitado[fila_ini][col_ini]:
                cola = [(fila_ini, col_ini)]
                visitado[fila_ini][col_ini] = True
                pixeles = []
                cabeza = 0
                while cabeza < len(cola):
                    f, c = cola[cabeza]
                    cabeza += 1
                    pixeles.append((f, c))
                    for df, dc in vecinos_4:
                        nf, nc = f + df, c + dc
                        if 0 <= nf < alto and 0 <= nc < ancho:
                            if imagen_binaria[nf, nc] == 255 and not visitado[nf][nc]:
                                visitado[nf][nc] = True
                                cola.append((nf, nc))

                area = len(pixeles)
                if area < min_area:
                    continue
                if max_area is not None and max_area > 0 and area > max_area:
                    continue

                fila_min = pixeles[0][0]
                fila_max = pixeles[0][0]
                col_min  = pixeles[0][1]
                col_max  = pixeles[0][1]
                for f, c in pixeles:
                    if f < fila_min:
                        fila_min = f
                    if f > fila_max:
                        fila_max = f
                    if c < col_min:
                        col_min = c
                    if c > col_max:
                        col_max = c

                perimetro = 0
                for f, c in pixeles:
                    es_borde = False
                    for df, dc in vecinos_4:
                        nf, nc = f + df, c + dc
                        if nf < 0 or nf >= alto or nc < 0 or nc >= ancho:
                            es_borde = True
                            break
                        if imagen_binaria[nf, nc] == 0:
                            es_borde = True
                            break
                    if es_borde:
                        perimetro += 1

                label += 1
                alto_bbox = fila_max - fila_min + 1
                ancho_bbox = col_max - col_min + 1
                regiones.append({
                    "label":     label,
                    "area":      area,
                    "perimetro": perimetro,
                    "bbox":      (fila_min, col_min, fila_max, col_max),
                    "alto":      alto_bbox,
                    "ancho":     ancho_bbox,
                    "pixeles":   pixeles,
                })

    for i in range(len(regiones) - 1):
        for j in range(i + 1, len(regiones)):
            if regiones[j]["area"] > regiones[i]["area"]:
                regiones[i], regiones[j] = regiones[j], regiones[i]

    print(f"[modelo] Regiones encontradas: {len(regiones)}")
    return regiones


def limpiar_mascara_regiones(imagen_binaria):
    """
    Suaviza el mapa de bordes antes de buscar componentes.
    Se conserva un píxel blanco solo si pertenece a una pequeña vecindad de borde.
    Ejemplo corto: un punto blanco aislado se borra; una línea de borde se conserva.
    """
    alto, ancho = imagen_binaria.shape
    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            if imagen_binaria[fila, columna] != 255:
                continue

            vecinos_blancos = 0
            for df in range(-1, 2):
                for dc in range(-1, 2):
                    if df == 0 and dc == 0:
                        continue
                    nf = fila + df
                    nc = columna + dc
                    if 0 <= nf < alto and 0 <= nc < ancho:
                        if imagen_binaria[nf, nc] == 255:
                            vecinos_blancos += 1

            if vecinos_blancos >= 2:
                resultado[fila, columna] = 255

    return resultado


def filtrar_regiones_utiles(regiones, alto, ancho, max_area=None):
    """
    Descarta componentes que suelen venir del marco de la placa o del ruido fino.
    La idea es que el clasificador reciba objetos compactos, no el contorno entero.
    Ejemplo corto: una caja enorme del borde de la placa se descarta, pero una letra
    con proporción razonable puede quedar como candidata.
    """
    total_pixeles = alto * ancho
    filtradas = []
    regiones_caracter = []

    for region in regiones:
        f0, c0, f1, c1 = region["bbox"]
        alto_bbox = f1 - f0 + 1
        ancho_bbox = c1 - c0 + 1
        area_bbox = alto_bbox * ancho_bbox
        area = region["area"]
        centro_fila = (f0 + f1) / 2.0
        relacion = ancho_bbox / alto_bbox if alto_bbox > 0 else 0

        if alto_bbox < 5 or ancho_bbox < 3:
            continue
        if max_area is None and area_bbox > total_pixeles * 0.18:
            continue
        if ancho_bbox > ancho * 0.55:
            continue
        if alto_bbox > alto * 0.80:
            continue
        if area_bbox > 0 and area / area_bbox < 0.02:
            continue

        filtradas.append(region)
        if (
            centro_fila >= alto * 0.36
            and centro_fila <= alto * 0.92
            and alto_bbox >= alto * 0.12
            and alto_bbox <= alto * 0.65
            and relacion >= 0.12
            and relacion <= 1.35
        ):
            regiones_caracter.append(region)

    if not filtradas:
        return regiones

    candidatas = regiones_caracter if regiones_caracter else filtradas
    if len(candidatas) > 16:
        candidatas = seleccionar_regiones_mas_relevantes(candidatas, 16)

    ordenar_regiones_lectura(candidatas)
    return candidatas


def seleccionar_regiones_mas_relevantes(regiones, limite):
    """Conserva las regiones con mayor área cuando todavía hay demasiados candidatos."""
    seleccionadas = []
    usadas = [False] * len(regiones)

    while len(seleccionadas) < limite and len(seleccionadas) < len(regiones):
        mejor_indice = -1
        mejor_area = -1
        for indice in range(len(regiones)):
            if usadas[indice]:
                continue
            area = regiones[indice]["area"]
            if area > mejor_area:
                mejor_area = area
                mejor_indice = indice

        if mejor_indice == -1:
            break
        usadas[mejor_indice] = True
        seleccionadas.append(regiones[mejor_indice])

    return seleccionadas


def ordenar_regiones_lectura(regiones):
    """
    Ordena caracteres de placa de izquierda a derecha.
    Ejemplo corto: si se detectan A, 3, B desordenados por área, se reacomodan por columna.
    """
    for i in range(1, len(regiones)):
        actual = regiones[i]
        posicion = i - 1
        fila_actual, col_actual = actual["bbox"][0], actual["bbox"][1]
        while posicion >= 0:
            fila_prev, col_prev = regiones[posicion]["bbox"][0], regiones[posicion]["bbox"][1]
            if col_prev < col_actual or (col_prev == col_actual and fila_prev <= fila_actual):
                break
            regiones[posicion + 1] = regiones[posicion]
            posicion -= 1
        regiones[posicion + 1] = actual


def dibujar_bounding_boxes(imagen_binaria, regiones):
    """
    Dibuja bounding boxes rojos sobre una copia RGB de la imagen binaria.
    Ejemplo corto: una región con bbox (10, 20, 40, 35) recibe un rectángulo alrededor.
    """
    alto, ancho = imagen_binaria.shape
    rgb = np.zeros((alto, ancho, 3), dtype=np.uint8)
    for fila in range(alto):
        for columna in range(ancho):
            v = imagen_binaria[fila, columna]
            rgb[fila, columna, 0] = v
            rgb[fila, columna, 1] = v
            rgb[fila, columna, 2] = v

    for reg in regiones:
        f0, c0, f1, c1 = reg["bbox"]
        for c in range(c0, c1 + 1):
            if 0 <= f0 < alto:
                rgb[f0, c] = [255, 0, 0]
            if 0 <= f1 < alto:
                rgb[f1, c] = [255, 0, 0]
        for f in range(f0, f1 + 1):
            if 0 <= c0 < ancho:
                rgb[f, c0] = [255, 0, 0]
            if 0 <= c1 < ancho:
                rgb[f, c1] = [255, 0, 0]

    return rgb


def dibujar_bounding_boxes_sobre_imagen(imagen_base, regiones):
    """Dibuja bounding boxes verdes sobre una imagen base RGB o gris."""
    if imagen_base.ndim == 2:
        alto, ancho = imagen_base.shape
        rgb = np.zeros((alto, ancho, 3), dtype=np.uint8)
        for fila in range(alto):
            for columna in range(ancho):
                valor = imagen_base[fila, columna]
                rgb[fila, columna, 0] = valor
                rgb[fila, columna, 1] = valor
                rgb[fila, columna, 2] = valor
    else:
        rgb = imagen_base.copy()

    alto, ancho = rgb.shape[:2]
    color = [80, 255, 80]
    grosor = 2

    for reg in regiones:
        f0, c0, f1, c1 = reg["bbox"]
        for offset in range(grosor):
            ff0 = f0 + offset
            ff1 = f1 - offset
            cc0 = c0 + offset
            cc1 = c1 - offset
            for c in range(max(0, cc0), min(ancho, cc1 + 1)):
                if 0 <= ff0 < alto:
                    rgb[ff0, c] = color
                if 0 <= ff1 < alto:
                    rgb[ff1, c] = color
            for f in range(max(0, ff0), min(alto, ff1 + 1)):
                if 0 <= cc0 < ancho:
                    rgb[f, cc0] = color
                if 0 <= cc1 < ancho:
                    rgb[f, cc1] = color

    return rgb


def extraer_recorte_gris(imagen, bbox):
    """Copia una región rectangular para aislar el objeto detectado."""
    f0, c0, f1, c1 = bbox
    alto = f1 - f0 + 1
    ancho = c1 - c0 + 1
    recorte = np.zeros((alto, ancho), dtype=np.uint8)

    for fila in range(alto):
        for columna in range(ancho):
            recorte[fila, columna] = imagen[f0 + fila, c0 + columna]

    return recorte


def redimensionar_vecino_mas_cercano(imagen, nuevo_alto, nuevo_ancho):
    """
    Redimensiona con vecino más cercano usando proporciones enteras.
    Ejemplo corto: un recorte de 20x30 puede llevarse a 64x64 sin crear tonos nuevos.
    """
    alto, ancho = imagen.shape
    resultado = np.zeros((nuevo_alto, nuevo_ancho), dtype=np.uint8)

    if alto == 0 or ancho == 0:
        return resultado

    for fila in range(nuevo_alto):
        origen_fila = int(fila * alto / nuevo_alto)
        if origen_fila >= alto:
            origen_fila = alto - 1
        for columna in range(nuevo_ancho):
            origen_columna = int(columna * ancho / nuevo_ancho)
            if origen_columna >= ancho:
                origen_columna = ancho - 1
            resultado[fila, columna] = imagen[origen_fila, origen_columna]

    return resultado


def extraer_recortes_normalizados(imagen, regiones, tamano=64):
    """
    Genera recortes individuales listos para clasificacion.
    Cada recorte conserva su bbox original y una imagen binaria tamano x tamano.
    Ejemplo corto: cada número detectado en la placa termina como un crop 64x64.
    """
    recortes = []

    for indice, region in enumerate(regiones, start=1):
        recorte = extraer_recorte_gris(imagen, region["bbox"])
        if recorte.shape[0] == 0 or recorte.shape[1] == 0:
            continue

        recorte_normalizado = redimensionar_vecino_mas_cercano(recorte, tamano, tamano)
        recortes.append({
            "indice": indice,
            "id_region": region["label"],
            "bbox": region["bbox"],
            "area": region["area"],
            "perimetro": region["perimetro"],
            "imagen": recorte_normalizado,
        })

    return recortes


def preparar_entrada_red_neuronal(recortes):
    """
    Convierte recortes 64x64 en vectores 0..1.
    Ejemplo corto: un píxel negro aporta 0.0 y uno blanco aporta 1.0 al vector.
    """
    entradas = []

    for recorte in recortes:
        imagen = recorte["imagen"]
        alto, ancho = imagen.shape
        vector = []

        for fila in range(alto):
            for columna in range(ancho):
                vector.append(int(imagen[fila, columna]) / 255.0)

        entradas.append({
            "indice": recorte["indice"],
            "id_region": recorte["id_region"],
            "bbox": recorte["bbox"],
            "area": recorte["area"],
            "alto": alto,
            "ancho": ancho,
            "vector": vector,
        })

    return entradas


def componer_tira_recortes(imagen, regiones, tamano=64, separacion=6):
    """Compone los recortes en un mosaico para revisar visualmente lo detectado."""
    recortes_info = extraer_recortes_normalizados(imagen, regiones, tamano)
    if not recortes_info:
        return np.zeros((tamano, tamano, 3), dtype=np.uint8)

    recortes = []
    for recorte_info in recortes_info:
        recortes.append(recorte_info["imagen"])

    if not recortes:
        return np.zeros((tamano, tamano, 3), dtype=np.uint8)

    columnas = 10
    if len(recortes) < columnas:
        columnas = len(recortes)
    filas = len(recortes) // columnas
    if len(recortes) % columnas != 0:
        filas += 1

    alto_total = filas * tamano + (filas - 1) * separacion
    ancho_total = columnas * tamano + (columnas - 1) * separacion
    mosaico = np.zeros((alto_total, ancho_total), dtype=np.uint8)
    for fila in range(alto_total):
        for columna in range(ancho_total):
            mosaico[fila, columna] = 245

    for indice in range(len(recortes)):
        recorte = recortes[indice]
        fila_base = indice // columnas
        columna_base = indice % columnas
        inicio_fila = fila_base * (tamano + separacion)
        inicio_columna = columna_base * (tamano + separacion)
        for fila in range(tamano):
            for columna in range(tamano):
                mosaico[inicio_fila + fila, inicio_columna + columna] = recorte[fila, columna]

    rgb = np.zeros((alto_total, ancho_total, 3), dtype=np.uint8)
    for fila in range(alto_total):
        for columna in range(ancho_total):
            valor = mosaico[fila, columna]
            rgb[fila, columna, 0] = valor
            rgb[fila, columna, 1] = valor
            rgb[fila, columna, 2] = valor
    return rgb

# Procesamiento Digital de Imágenes

Aplicación de escritorio en Python para probar filtros de procesamiento digital de imágenes en dominio espacial y en dominio de frecuencia. El flujo está pensado para analizar placas vehiculares: preparar la imagen, resaltar bordes, detectar regiones con bounding boxes y dejar recortes normalizados que luego puedan conectarse a una red neuronal.

## Qué hace

El sistema permite cargar una imagen RGB y procesarla por etapas:

1. Conversión manual a escala de grises.
2. Ruido sal y pimienta configurable.
3. Normalización por ecualización de histograma.
4. Suavizado espacial o frecuencial.
5. Acentuado espacial o frecuencial.
6. Binarización por umbral.
7. Detección de bordes por gradiente.
8. Limpieza de máscara y detección de regiones.
9. Bounding boxes sobre la imagen original.
10. Recortes normalizados para clasificación.

La aplicación permite aplicar el pipeline completo o actualizar etapas específicas para comparar resultados sin recalcular todo cada vez.

## Filtros disponibles

### Suavizado espacial

- Media
- Mediana
- Moda

### Suavizado en frecuencia

- Ideal
- Gaussiano
- Butterworth

### Acentuado espacial

- Laplaciano
- Pasa-alto
- High-Boost

### Acentuado en frecuencia

- Ideal
- Gaussiano
- Butterworth
- High-Boost

### Gradientes

- Roberts
- Prewitt
- Sobel
- Kirsch
- Laplaciano

## Resultados finales

Después de aplicar la detección de regiones, el sistema genera:

- Gradiente final.
- Gradiente binario.
- Máscara filtrada para regiones.
- Imagen con bounding boxes.
- Mosaico de recortes detectados.
- Recortes individuales `64x64`.
- Vectores normalizados `0..1` para una red neuronal futura.

Desde la interfaz se puede usar el botón **Guardar final** para exportar:

```text
01_gradiente.png
02_gradiente_binario.png
03_mascara_regiones.png
04_bounding_boxes.png
05_recortes_mosaico.png
recortes_clasificador/*.png
recortes_clasificador/metadata_recortes.csv
```

## Estructura del proyecto

```text
.
├── main.py
├── requirements.txt
├── src
│   ├── aplicacion.py
│   ├── modelo
│   │   ├── base.py
│   │   ├── frecuencia.py
│   │   ├── gradientes.py
│   │   ├── regiones.py
│   │   ├── suavizado.py
│   │   └── utils.py
│   └── ui
│       ├── controles.py
│       ├── exportacion.py
│       ├── procesamiento.py
│       ├── tabs.py
│       ├── ventana.py
│       └── widgets.py
└── placas
```

`main.py` solo abre la aplicación. La lógica del procesamiento está en `src/modelo`, el pipeline en `src/aplicacion.py` y la interfaz en `src/ui`.

## Instalación

Se recomienda usar un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

Si no activaste el entorno virtual:

```bash
.venv/bin/python main.py
```

## Sobre la implementación

Los algoritmos principales están escritos con recorridos explícitos sobre píxeles. NumPy se usa como estructura de matriz y para la FFT; OpenCV se usa para cargar y guardar imágenes; Matplotlib se usa para mostrar resultados en la interfaz.

La parte de frecuencia utiliza `numpy.fft`, pero las máscaras ideal, gaussiana, Butterworth, pasa-altas y high-boost se construyen dentro del proyecto. La detección de regiones se hace con BFS manual, sin `connectedComponents`.

## Nota para clasificación futura

El pipeline ya deja preparada la salida para una red neuronal mediante `entradas_red_neuronal`: una lista de vectores normalizados a partir de los recortes `64x64`. Cada entrada conserva metadatos como región, bounding box y área para poder rastrear de dónde salió cada carácter.

# Procesamiento de Imagenes

Aplicacion de escritorio en Python para comparar procesamiento en dominio espacial y dominio de frecuencia.

## Flujo

1. Carga de imagen RGB
2. Conversion manual a escala de grises
3. Normalizacion (ecualizacion) manual de histograma
4. Ruido sal y pimienta (manual, sin imnoise)
5. Filtros espaciales (dominio temporal): media, mediana y moda
6. Filtros de gradiente / pasa altos: Roberts, Prewitt, Sobel, Laplaciano
7. Filtro gaussiano pasa bajas en dominio de frecuencia (FFT + mascara gaussiana manual)

## Librerias

- `OpenCV`: carga de imagen y conversion inicial BGR a RGB
- `NumPy`: manejo de matrices y FFT
- `PySide6`: interfaz grafica
- `Matplotlib`: visualizacion en la UI

## Ejecutar

```bash
pip install -r requirements.txt
python main.py
```

## Nota

Los filtros espaciales principales estan hechos con recorridos manuales. La parte de frecuencia usa la FFT de `NumPy` y una mascara gaussiana construida en el codigo.

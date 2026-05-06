# Procesamiento de Imagenes

Aplicacion de escritorio en Python para comparar procesamiento en dominio espacial y dominio de frecuencia.

## Flujo

- Carga de imagen RGB
- Conversion manual a escala de grises
- Ecualizacion manual de histograma
- Binarizacion por umbral
- Ruido sal y pimienta
- Filtros espaciales: media, mediana y moda
- Filtro gaussiano pasa bajas en frecuencia

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

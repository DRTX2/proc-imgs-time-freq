# Herramienta de Preprocesamiento de Imágenes

Aplicación de escritorio construida con Python y PySide6 para preprocesamiento de imágenes.

## Funcionalidades

1. Normalización min-max por canal RGB con visualización de histogramas.
2. Conversión a escala de grises (luma).
3. Compresión por bloques ($N \times N$) mediante promedio.
4. Binarización por umbral.

## Requisitos

- Python 3.9 o superior.

## Instalación

### 1. Crear y activar entorno virtual

Windows:
```cmd
python -m venv env
.\env\Scripts\activate
```

Linux / macOS:
```bash
python3 -m venv env
source env/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

## Notas

- La aplicación intenta cargar una imagen inicial por defecto desde `informe/formación estelar de W51.png`.
- En Windows, la carga de imágenes usa lectura por bytes para evitar problemas con rutas que contienen caracteres especiales.

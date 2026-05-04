import numpy as np
import modelo


class EstadoFiltrado:
    """
    Mantiene el estado de la aplicación: imagen cargada, imagen con ruido,
    y los resultados de aplicar cada filtro.
    """

    def __init__(self):
        # Imagen base en negro para iniciar la interfaz sin depender de archivos.
        base = np.zeros((320, 480, 3), dtype=np.uint8)

        self.image_path = None
        self.img_rgb = base.copy()
        self.img_gris = modelo.rgb_a_gris(base)
        self.img_ruido = self.img_gris.copy()

        # Parámetros de ruido
        self.prob_ruido = 0.05

        # Parámetros de filtro espacial
        self.tipo_filtro_espacial = "media"  # media | mediana | moda
        self.tam_kernel = 3
        self.resultado_espacial = self.img_gris.copy()

        # Parámetros de filtro en frecuencia
        self.radio_frecuencia = 30
        self.resultado_frecuencia = self.img_gris.copy()
        self.espectro_magnitud = self.img_gris.copy()

        # Binarización (threshold = media)
        self.resultado_binario = self.img_gris.copy()
        self.umbral_bin = 128

    def establecer_imagen(self, ruta):
        """Carga una nueva imagen y recalcula la versión en grises."""
        self.image_path = ruta
        self.img_rgb = modelo.cargar_imagen(ruta)
        self.img_gris = modelo.rgb_a_gris(self.img_rgb)
        self.aplicar_ruido()

    def aplicar_ruido(self, probabilidad=None):
        """Agrega ruido sal y pimienta a la imagen en grises."""
        if probabilidad is not None:
            self.prob_ruido = probabilidad

        self.img_ruido = modelo.ruido_sal_pimienta(self.img_gris, self.prob_ruido)

    def aplicar_filtro_espacial(self, tipo=None, tam_kernel=None):
        """
        Aplica el filtro espacial seleccionado sobre la imagen con ruido.
        tipo: 'media', 'mediana' o 'moda'
        """
        if tipo is not None:
            self.tipo_filtro_espacial = tipo
        if tam_kernel is not None:
            self.tam_kernel = tam_kernel

        if self.tipo_filtro_espacial == "media":
            self.resultado_espacial = modelo.filtro_media(
                self.img_ruido, self.tam_kernel
            )
        elif self.tipo_filtro_espacial == "mediana":
            self.resultado_espacial = modelo.filtro_mediana(
                self.img_ruido, self.tam_kernel
            )
        elif self.tipo_filtro_espacial == "moda":
            self.resultado_espacial = modelo.filtro_moda(
                self.img_ruido, self.tam_kernel
            )

        # Actualizar binarización con la media como umbral
        self.umbral_bin = int(modelo.calcular_media(self.resultado_espacial))
        self.resultado_binario = modelo.binarizar(
            self.resultado_espacial, self.umbral_bin
        )

    def aplicar_filtro_frecuencia(self, radio=None):
        """
        Aplica filtro pasabajas en el dominio de frecuencia sobre la imagen
        con ruido. Usa la FFT y la FFT inversa con una máscara circular.
        """
        if radio is not None:
            self.radio_frecuencia = radio

        self.espectro_magnitud = modelo.obtener_espectro_magnitud(self.img_ruido)
        self.resultado_frecuencia = modelo.filtro_frecuencia_pasabajas(
            self.img_ruido, self.radio_frecuencia
        )

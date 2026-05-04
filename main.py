import sys
import signal
from PySide6.QtWidgets import QApplication
from interfaz import VentanaPrincipal
from estado import EstadoFiltrado
import modelo


def main():
    # Permitir que Ctrl+C cierre la aplicación correctamente
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)

    # Crear el estado de la aplicación
    estado = EstadoFiltrado()

    # Buscar imagen de prueba si existe
    ruta_inicial = modelo.buscar_imagen_inicial()
    if ruta_inicial:
        estado.establecer_imagen(ruta_inicial)

    # Crear la ventana e inyectar el estado
    ventana = VentanaPrincipal(estado)
    ventana.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

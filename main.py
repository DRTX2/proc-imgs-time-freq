import sys
from PySide6.QtWidgets import QApplication
from interfaz import VentanaPrincipal

def main():
    app = QApplication(sys.argv)
    
    # Crear y mostrar la ventana principal
    ventana = VentanaPrincipal()
    ventana.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

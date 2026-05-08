import sys
from PySide6.QtWidgets import QApplication
from src.ui import VentanaPrincipal

def main():
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

"""Estilos visuales de la interfaz PySide."""

def estilo_slider(color):
    return f"""
    QSlider::groove:horizontal {{
        border: none;
        height: 6px;
        background: #A0A7B4;
        border-radius: 4px;
    }}
    QSlider::sub-page:horizontal {{
        background: {color};
        border-radius: 4px;
    }}
    QSlider::add-page:horizontal {{
        background: #A0A7B4;
        border-radius: 4px;
    }}
    QSlider::handle:horizontal {{
        background: white;
        border: 2px solid {color};
        width: 14px;
        margin: -5px 0;
        border-radius: 9px;
    }}
    """


APP_STYLE = """
QMainWindow { background: #EEF2F7; }
QWidget { color: #253040; font-size: 13px; }
QFrame#Sidebar {
    background: #15202B;
    border: none;
}
QFrame#Panel {
    background: white;
    border: 1px solid #D8DCE4;
    border-radius: 16px;
}
QFrame#ControlCard {
    background: #1D2B38;
    border: 1px solid #314353;
    border-radius: 18px;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollArea#TabScrollArea {
    background: white;
}
QWidget#TabScrollContent {
    background: white;
}
QScrollBar:vertical {
    background: #15202B;
    width: 10px;
    margin: 6px 0;
}
QScrollBar::handle:vertical {
    background: #304252;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QTabWidget::pane {
    border: 1px solid #D8DCE4;
    background: white;
    border-radius: 14px;
}
QTabBar::tab {
    background: #DDE7F1;
    color: #304055;
    border: 1px solid #C7D4E1;
    padding: 10px 16px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 700;
}
QTabBar::tab:selected {
    background: white;
    border-bottom-color: white;
}
QLabel#MainTitle {
    font-size: 18px;
    font-weight: 800;
    color: #222B38;
}
QLabel#SidebarTitle {
    font-size: 18px;
    font-weight: 800;
    color: #F5F7FA;
}
QLabel#SidebarText {
    color: #AFC1D1;
    line-height: 1.3;
}
QLabel#SectionTitle {
    font-size: 13px;
    font-weight: 700;
    color: #2F3B4C;
}
QLabel#Muted {
    color: #5F6C7E;
}
QLabel#SidebarMuted {
    color: #9EB1C4;
}
QLabel#CardTitle {
    font-size: 14px;
    font-weight: 800;
    color: #F4F7FB;
}
QLabel#CardHint {
    color: #95A8BA;
    line-height: 1.25;
}
QLabel#ValueBadge {
    background: #2B4053;
    color: #F7FBFF;
    border-radius: 10px;
    padding: 5px 10px;
    font-weight: 700;
}
QPushButton {
    background: #E56B4A;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 24px;
    padding: 10px 12px;
    font-weight: 700;
}
QPushButton:hover { background: #D55D3C; }
QPushButton:disabled { background: #9EA8B4; }
QPushButton#ActionButton {
    background: #24A187;
}
QPushButton#ActionButton:hover {
    background: #1D8A73;
}
QPushButton#SecondaryButton {
    background: #357ABD;
}
QPushButton#SecondaryButton:hover {
    background: #2D68A3;
}
QPushButton#DangerButton {
    background: #C24E4E;
}
QPushButton#DangerButton:hover {
    background: #AA4242;
}
QPushButton[compact="true"] {
    padding: 10px 6px;
    font-size: 12px;
}
QComboBox {
    background: #243648;
    color: #F0F5FA;
    border: 2px solid #4A7FA5;
    border-radius: 10px;
    min-height: 20px;
    padding: 8px 9px;
    font-weight: 700;
}
QComboBox::drop-down {
    color: white;
    border: none;
}
QComboBox QAbstractItemView {
    background: #1D2B38;
    color: #F0F5FA;
    selection-background-color: #357ABD;
    selection-color: white;
    border: 1px solid #4A7FA5;
    outline: none;
}
QSpinBox {
    background: #243648;
    color: #F4F8FC;
    border: 1px solid #4A7FA5;
    border-radius: 6px;
    padding: 4px 22px 4px 8px;
    min-height: 18px;
}
QSpinBox:disabled {
    color: #8EA1B3;
    background: #172331;
}
QSpinBox::up-button, QSpinBox::down-button {
    background: #1B2A38;
    border-left: 1px solid #3C6C8E;
    width: 18px;
    subcontrol-origin: border;
}
QSpinBox::up-button {
    subcontrol-position: top right;
    border-top-right-radius: 5px;
}
QSpinBox::down-button {
    subcontrol-position: bottom right;
    border-bottom-right-radius: 5px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background: #27465D;
}
QSpinBox::up-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #BFD7EA;
    width: 0px;
    height: 0px;
}
QSpinBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #BFD7EA;
    width: 0px;
    height: 0px;
}
"""

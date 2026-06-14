from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication
import sys
from NameSelectorWidget import NameSelectorWidget


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Demo")

        self.label = QLabel("Please select a name")
        self.name_selector = NameSelectorWidget()

        self.name_selector.name_selected.connect(self.handle_name_selected)

        layout = QVBoxLayout()
        layout.addWidget(self.name_selector)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def handle_name_selected(self, name):
        self.label.setText(f"Hello, {name}!")
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QComboBox
from PyQt5.QtCore import pyqtSignal


class NameSelectorWidget(QWidget):
    name_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.dropdown = QComboBox()
        self.dropdown.addItem("Choose a name...")
        self.dropdown.addItems(["Alice", "Bob", "Charlie"])

        layout = QVBoxLayout()
        layout.addWidget(self.dropdown)
        self.setLayout(layout)

        self.dropdown.currentIndexChanged.connect(self._on_index_changed)

    def _on_index_changed(self, index):
        if index == 0:
            return
        name = self.dropdown.currentText()
        self.name_selected.emit(name)


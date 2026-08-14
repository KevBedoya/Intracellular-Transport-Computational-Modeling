from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QGroupBox, QComboBox, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QUrl
# from PyQt5.QtGui import QTextCursor
import os
import subprocess


import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class PNGPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.dropdown = QComboBox()
        self.dropdown.currentIndexChanged.connect(self.update_image)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(600, 600)
        # self.image_label.setScaledContents(True)

        self.layout.addWidget(self.dropdown)
        self.layout.addWidget(self.image_label)

        self.png_paths = []
        # Cache of the currently-decoded original pixmap so window resizes
        # re-scale it instead of re-reading the PNG from disk every event.
        self._current_pixmap = None
        self._current_index = -1

    def update_png_list(self, png_paths):
        self.png_paths = png_paths or []
        # Invalidate the cache so a new result set always re-decodes.
        self._current_pixmap = None
        self._current_index = -1
        self.dropdown.clear()

        for path in self.png_paths:
            filename = os.path.basename(path)
            self.dropdown.addItem(filename)

        if self.png_paths:
            self.update_image(0)
        else:
            self.image_label.clear()

    def update_image(self, index):
        if not (0 <= index < len(self.png_paths)):
            return
        # Decode from disk only when the selected image actually changes.
        if index != self._current_index or self._current_pixmap is None:
            pixmap = QPixmap(self.png_paths[index])
            if pixmap.isNull():
                print(f"Warning: failed to load image {self.png_paths[index]}")
                return
            self._current_pixmap = pixmap
            self._current_index = index
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self):
        """Scale the cached original pixmap to the current label size."""
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return
        self.image_label.setPixmap(self._current_pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-scale from the cached pixmap; no disk read on resize.
        self._apply_scaled_pixmap()


class OutputFilesWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.csv_container = QWidget()
        self.csv_layout = QVBoxLayout()
        self.csv_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.csv_container.setLayout(self.csv_layout)

        self.csv_scroll = QScrollArea()
        self.csv_scroll.setWidgetResizable(True)
        self.csv_scroll.setWidget(self.csv_container)

        self.layout.addWidget(self.csv_scroll)

    @staticmethod
    def open_in_file_explorer(file_path):
        folder = os.path.dirname(file_path)
        if os.name == "nt":
            os.startfile(folder)
        elif os.name == "posix":
            if "Darwin" in os.uname().sysname:
                subprocess.run(["open", folder])
            else:
                subprocess.run(["xdg-open", folder])

    def update_display(self, csv_paths, png_paths=None):
        # Clear previous displays

        while self.csv_layout.count():
            item = self.csv_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for path in csv_paths:
            label = QLabel(f'<a href="{path}">{os.path.basename(path)}</a>')
            label.setTextInteractionFlags(Qt.TextBrowserInteraction)
            label.setOpenExternalLinks(False)
            label.linkActivated.connect(lambda file_path=path: self.open_in_file_explorer(file_path))
            label.setStyleSheet("padding: 4px;")
            self.csv_layout.addWidget(label)
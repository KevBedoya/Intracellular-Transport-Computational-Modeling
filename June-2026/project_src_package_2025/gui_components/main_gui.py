import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QComboBox, QPushButton, QMessageBox, QStatusBar, QSizeGrip)
from PyQt5.QtGui import QGuiApplication, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QPoint
from .control_panel import ControlPanel
from . import fp
from . import theme

from project_src_package_2025.gui_components import styles


class GridOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.mouse_pos = QPoint()
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnBottomHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent")
        self.step = 40

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor(200, 200, 200, 100))
        pen.setWidth(1)
        painter.setPen(pen)

        width = self.width()
        height = self.height()

        for x in range(0, width, self.step):
            painter.drawLine(x, 0, x, height)
            painter.drawText(x + 2, 12, str(x))

        for y in range(0, height, self.step):
            painter.drawLine(0, y, width, y)
            painter.drawText(2, y + 12, str(y))

        painter.setPen(QPen(QColor(255, 0, 0, 200)))
        x, y = self.mouse_pos.x(), self.mouse_pos.y()
        painter.drawText(x + 10, y - 10, f"({x}, {y})")
        painter.drawEllipse(self.mouse_pos, 3, 3)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        debug_mode = False

        # --- SIZE & POSITION: use a normal window size, center it, and keep it movable ---
        screen = QGuiApplication.primaryScreen()
        ag = screen.availableGeometry()

        # (no change) choose a reasonable starting size
        self.resize(int(ag.width() * 0.7), int(ag.height() * 0.7))  # e.g., 70% of screen

        # center on screen
        frame = self.frameGeometry()
        frame.moveCenter(ag.center())
        self.move(frame.topLeft())

        # ensure not maximized/fullscreen
        # self.setWindowState(Qt.WindowMaximized)  # <-- COMMENT OUT if you had this anywhere
        self.setWindowState(Qt.WindowNoState)  # keep window freely movable/resizable

        # self.setFixedSize(self.size())  # <-- COMMENT OUT if present anywhere
        self.setMinimumSize(400, 300)  # sensible floor
        # optional: no upper bound so corners can drag freely
        # self.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)  # needs from PyQt5.QtCore import QWIDGETSIZE_MAX

        self.setWindowTitle("Biophysics Software (August-2025)")
        """ 
            Potential names:
                'Statistical computing software for intracellular transport dynamics under microtubule morphologies" 
                'Discrete computational modeling of intracellular transport dynamics under microtubule morphologies"
                'Computational Modeling of Intracellular Transport Software' 
        """

        # --- CENTRAL WIDGET + LAYOUT WITH MARGINS ---
        container = QWidget(self)
        layout = QVBoxLayout(container)

        # Set nice inner margins (L, T, R, B) and spacing between widgets
        layout.setContentsMargins(16, 16, 16, 16)  # tweak to taste
        layout.setSpacing(12)

        # container.setContentsMargins(16, 16, 16, 16)

        # Your control panel
        self.control_panel = ControlPanel(self)
        layout.addWidget(self.control_panel)

        self.setCentralWidget(container)

        # --- ADD A VISIBLE RESIZE GRIP (bottom-right) ---
        status = QStatusBar(self)
        self.setStatusBar(status)
        status.setSizeGripEnabled(False)  # use an explicit grip so it's always visible
        grip = QSizeGrip(status)
        status.addPermanentWidget(grip, 0)

        if debug_mode:
            self.grid_overlay = GridOverlay(self)
            self.grid_overlay.resize(self.size())
            self.grid_overlay.show()

    def resizeEvent(self, event):
        if hasattr(self, "grid_overlay"):
            self.grid_overlay.resize(self.size())
        super().resizeEvent(event)


def run_app():
    # HiDPI attributes must be set before the QApplication is constructed so the
    # interface stays crisp on Retina / scaled displays.
    theme.enable_high_dpi()
    app = QApplication(sys.argv)

    # Set the base font up front so widget size hints use the final metrics
    # (prevents clipped tab labels / button rows), then apply the stylesheet
    # (importlib.resources with a filesystem fallback for frozen builds).
    theme.apply_base_font(app)
    app.setStyleSheet(theme.load_stylesheet())

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
